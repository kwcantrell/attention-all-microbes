from __future__ import annotations

from math import log
from typing import Optional, Union

import tensorflow as tf
import tensorflow_models as tfm

from aam.losses import PairwiseLoss, _pairwise_distances
from aam.models.multihead_attention_pooling import MultiHeadAttentionPooling
from aam.models.transformers import TransformerEncoder
from aam.models.unifrac_denoising import UnifracDenoiser
from aam.models.unifrac_encoder import UnifracEncoder
from aam.models.utils import sort_using_counts, to_batch
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler
from aam.utils import create_random_mask, float_mask


@tf.keras.saving.register_keras_serializable(package="SequenceRegressor")
class SequenceRegressor(tf.keras.Model):
    def __init__(
        self,
        token_limit: int,
        base_output_dim: Optional[int] = None,
        shift: float = 0.0,
        scale: float = 1.0,
        dropout_rate: float = 0.0,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 1024,
        intermediate_activation: str = "relu",
        base_model: UnifracDenoiser = None,
        freeze_base: bool = False,
        penalty: float = 1.0,
        nuc_penalty: float = 1.0,
        max_bp: int = 150,
        is_16S: bool = True,
        vocab_size: int = 6,
        out_dim: int = 1,
        classifier: bool = False,
        add_token: bool = True,
        class_weights: list = None,
        asv_dropout_rate: float = 0.0,
        accumulation_steps: int = 1,
        scale_losses=False,
        normalize_outputs=False,
        use_residual_connections=True,
        include_count_encoder=True,
        use_linear_bias=True,
        **kwargs,
    ):
        super(SequenceRegressor, self).__init__(**kwargs)
        self.token_limit = token_limit
        self.base_output_dim = base_output_dim
        self.shift = shift
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.freeze_base = freeze_base
        self.penalty = penalty
        self.nuc_penalty = nuc_penalty
        self.max_bp = max_bp
        self.is_16S = is_16S
        self.vocab_size = vocab_size
        self.out_dim = out_dim
        self.classifier = classifier
        self.add_token = add_token
        self.class_weights = class_weights
        self.asv_dropout_rate = asv_dropout_rate
        self.accumulation_steps = accumulation_steps
        self.scale_losses = scale_losses
        self.normalize_outputs = normalize_outputs
        self.use_residual_connections = use_residual_connections
        self.include_count_encoder = include_count_encoder
        self.use_linear_bias = use_linear_bias
        self.loss_tracker = tf.keras.metrics.Mean()

        # layers used in model
        self.combined_base = False
        self.base_model = base_model
        self.base_model.trainable = False
        self.embedding_loss = PairwiseLoss(use_mean_pairs=False)
        self.embedding_tracker = tf.keras.metrics.Mean()

        self.target_tracker = tf.keras.metrics.Mean()
        if not self.classifier:
            self.metric_tracker = tf.keras.metrics.MeanAbsoluteError()
            self.metric_string = "mae"
        else:
            self.metric_tracker = tf.keras.metrics.SparseCategoricalAccuracy()
            self.metric_string = "accuracy"

    def build(self, input_shape):
        if self.built:
            return

        if self.freeze_base:
            print("Freezing base model...")
            self.base_model.trainable = False

        def _ff_block(output_dim, use_bias=True, dropout_rate=None):
            block = [
                tf.keras.layers.LayerNormalization(dtype=tf.float32),
                tf.keras.layers.Dense(
                    output_dim,
                    use_bias=use_bias,
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                ),
                tf.keras.layers.LayerNormalization(dtype=tf.float32),
                tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x)),
            ]
            if dropout_rate:
                block.append(tf.keras.layers.Dropout(dropout_rate))
            return block

        self.sample_embedding_ff = tf.keras.Sequential(_ff_block(32, dropout_rate=0.5))
        self.tax_count_ff = tf.keras.Sequential(_ff_block(32, dropout_rate=0.5))
        self.out_ff = tf.keras.layers.Dense(
            self.out_dim, kernel_initializer=tf.keras.initializers.HeUniform()
        )
        self.output_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        super(SequenceRegressor, self).build(input_shape)

    def _compute_loss(
        self,
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        model_outputs: Union[
            tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        ],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        sample_embeddings, y_pred = model_outputs
        y_true = tf.reshape(tf.cast(y_true, tf.float32), shape=[-1, 1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), shape=[-1, 1])

        # step 1: minimize mse
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # step 2: pairwise distance of sample_embeddings should match pairwise distance of target
        y_true_dist = _pairwise_distances(y_true, squared=False)
        embedding_loss = tf.reduce_mean(
            self.embedding_loss(y_true_dist, sample_embeddings)
        )
        embedding_loss = 0.0
        return mse_loss + embedding_loss, mse_loss, embedding_loss

    def _compute_metric(
        self,
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: Union[
            tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        ],
    ):
        _, y_pred = outputs
        y_true = y_true * self.scale + self.shift
        y_pred = y_pred * self.scale + self.shift
        self.metric_tracker.update_state(y_true, y_pred)

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        y_true = y

        _, y_pred = self(inputs, training=False)

        if not self.classifier:
            y_true = y_true * self.scale + self.shift
            y_pred = y_pred * self.scale + self.shift
        else:
            y_true = tf.cast(y_true, dtype=tf.int32)
            y_pred = tf.argmax(tf.keras.activations.softmax(y_pred), axis=-1)
        return y_pred, y_true

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        y_target = y

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, target_loss, embedding_loss = self._compute_loss(y_target, outputs)
            if self.compute_dtype == "float16":
                print("Using scaled loss")
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.target_tracker.update_state(target_loss)
        self.embedding_tracker.update_state(embedding_loss)
        self._compute_metric(y_target, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "target_loss": self.target_tracker.result(),
            "embedding_loss": self.embedding_tracker.result(),
            self.metric_string: self.metric_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def test_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        y_target = y
        outputs = self(inputs, training=False)
        loss, target_loss, embedding_loss = self._compute_loss(y_target, outputs)

        self.loss_tracker.update_state(loss)
        self.target_tracker.update_state(target_loss)
        self.embedding_tracker.update_state(embedding_loss)
        self._compute_metric(y_target, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "target_loss": self.target_tracker.result(),
            "embedding_loss": self.embedding_tracker.result(),
            self.metric_string: self.metric_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def call(
        self, inputs, training: bool = False
    ) -> Union[
        tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        training = training and self.trainable
        if len(inputs) == 5:
            inputs, taxonomy_counts = inputs[:4], inputs[4]
            # asv_embeddings, counts = self.base_model.asv_embeddings(inputs)
            # mask = tf.cast(counts > 0, dtype=self.compute_dtype)
            # asv_embeddings = tf.cast(asv_embeddings, dtype=self.compute_dtype) * mask

            # sample_embeddings = tf.reduce_sum(asv_embeddings, axis=1) / tf.reduce_sum(
            #     mask, axis=1
            # )
            _, sample_embeddings = self.base_model(inputs, training=False)
            sample_embeddings = self.sample_embedding_ff(
                sample_embeddings, training=training
            )

            # compute relative abundance
            taxonomy_counts = tf.cast(taxonomy_counts, dtype=tf.float32)
            total_counts = tf.reduce_sum(taxonomy_counts, axis=1, keepdims=True)
            taxonomy_counts = taxonomy_counts / total_counts
            taxonomy_counts = self.tax_count_ff(taxonomy_counts, training=training)

            sample_embeddings = (sample_embeddings + taxonomy_counts) / 2.0
            output = self.out_ff(sample_embeddings)
            return self.output_activation(sample_embeddings), self.output_activation(
                output
            )
        else:
            asv_embeddings, counts = self.base_model.asv_embeddings(inputs)
            mask = tf.cast(counts > 0, dtype=self.compute_dtype)
            asv_embeddings = tf.cast(asv_embeddings, dtype=self.compute_dtype) * mask

            sample_embeddings = tf.reduce_sum(asv_embeddings, axis=1) / tf.reduce_sum(
                mask, axis=1
            )
            sample_embeddings = self.regressor(sample_embeddings, training=training)
            output = self.out_ff(sample_embeddings)
            return self.output_activation(sample_embeddings), self.output_activation(
                output
            )

    def get_config(self):
        config = super(SequenceRegressor, self).get_config()
        config.update(
            {
                "token_limit": self.token_limit,
                "base_output_dim": self.base_output_dim,
                "shift": self.shift,
                "scale": self.scale,
                "dropout_rate": self.dropout_rate,
                "embedding_dim": self.embedding_dim,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "intermediate_activation": self.intermediate_activation,
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "freeze_base": self.freeze_base,
                "penalty": self.penalty,
                "nuc_penalty": self.nuc_penalty,
                "max_bp": self.max_bp,
                "is_16S": self.is_16S,
                "vocab_size": self.vocab_size,
                "out_dim": self.out_dim,
                "classifier": self.classifier,
                "add_token": self.add_token,
                "class_weights": self.class_weights,
                "asv_dropout_rate": self.asv_dropout_rate,
                "accumulation_steps": self.accumulation_steps,
                "scale_losses": self.scale_losses,
                "normalize_outputs": self.normalize_outputs,
                "use_residual_connections": self.use_residual_connections,
                "build_input_shape": self.get_build_config(),
                "include_count_encoder": self.include_count_encoder,
                "use_linear_bias": self.use_linear_bias,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["base_model"] = tf.keras.saving.deserialize_keras_object(
            config["base_model"]
        )

        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)

        if input_shape is not None:
            model.build(input_shape)

        return model
