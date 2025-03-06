from __future__ import annotations

import json
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


@tf.keras.saving.register_keras_serializable(package="ConvolutionBlock")
class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, pool=False, **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool = pool

    def build(self, input_shape):
        self.conv_inner = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
        )
        if self.pool:
            self.conv_outer = tf.keras.layers.Conv1D(
                filters=self.filters * 2,
                kernel_size=self.kernel_size,
                strides=2,
                padding="same",
            )
            self.res_pool = tf.keras.layers.Conv1D(
                filters=self.filters * 2, kernel_size=1, strides=2, padding="same"
            )
        else:
            self.conv_outer = tf.keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=1,
                padding="same",
            )
        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )
        self.activation = tf.keras.layers.Activation("gelu")

    def call(self, inputs, training=False):
        output = self.conv_inner(inputs)
        output = self.activation(output)
        output = self.conv_outer(output)
        output = self.activation(output)
        if self.pool:
            inputs = self.res_pool(inputs)
        residual = inputs + self._rezero * output
        return residual

    def get_config(self):
        config = super(ConvolutionBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool": self.pool,
            }
        )
        return config


@tf.keras.saving.register_keras_serializable(package="SequenceRegressorV2")
class SequenceRegressorV2(tf.keras.Model):
    def __init__(
        self,
        hidden_dim: int,
        num_hidden_layers: int = 2,
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
        super(SequenceRegressorV2, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
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

        # layers used in model
        self.base_model = base_model
        if self.base_model is not None:
            self.base_model.trainable = False

    def build(self, input_shape):
        if self.built:
            return

        if self.freeze_base:
            print("Freezing base model...")
            if self.base_model is not None:
                self.base_model.trainable = False

        self.dropout_layers = []
        filters = 1
        ff_layers = [tf.keras.layers.Input([(input_shape[0])[-1], 1])]
        for _ in range(self.num_hidden_layers):
            ff_layers += [
                ConvolutionBlock(filters, 5, pool=False),
                ConvolutionBlock(filters, 5, pool=False),
                ConvolutionBlock(filters, 5, pool=False),
                ConvolutionBlock(filters, 5, pool=True),
            ]
            filters *= 2
        self.ff = tf.keras.Sequential(ff_layers + [tf.keras.layers.Flatten()])
        self.ff.build([(input_shape[0])[-1], 1])
        self.out_ff = tf.keras.layers.Dense(self.out_dim)
        super(SequenceRegressorV2, self).build(input_shape)

    def _compute_loss(
        self,
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        model_outputs: Union[
            tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        ],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        sample_embeddings, y_pred = model_outputs
        if isinstance(y_true, (list, tuple)):
            y_true, sample_weights = y_true
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
            y_pred = tf.argmax(
                tf.keras.activations.softmax(tf.cast(y_pred, dtype=tf.float64)), axis=-1
            )
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

    def batch_embeddings(
        self, asv_embeddings, batch_indicies, counts, asv_indices=None
    ):
        emb_dim = tf.shape(asv_embeddings)[-1]
        batch_indicies = tf.cast(batch_indicies, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)

        if asv_indices is not None:
            asv_embeddings = tf.gather(asv_embeddings, asv_indices)
        batch_shape = tf.reduce_max(batch_indicies[:, 0]) + 1
        max_unique = tf.reduce_max(batch_indicies[:, 1]) + 1
        batch_embeddings = tf.scatter_nd(
            batch_indicies, asv_embeddings, shape=[batch_shape, max_unique, emb_dim]
        )
        counts = tf.scatter_nd(
            batch_indicies, counts, shape=[batch_shape, max_unique, 1]
        )
        return batch_embeddings, counts

    def call(
        self, inputs, training: bool = False
    ) -> Union[
        tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        training = training and self.trainable
        if len(inputs) == 4:
            if self.base_model is not None:
                asv_embeddings, counts = self.base_model.asv_embeddings(inputs)
            else:
                tokens, batch_indices, asv_indices, counts = inputs
                asv_embeddings, counts = self.batch_embeddings(
                    tokens, batch_indices, counts, asv_indices
                )
            mask = tf.cast(counts > 0, dtype=self.compute_dtype)
            asv_embeddings = tf.cast(asv_embeddings, dtype=self.compute_dtype) * mask
        else:
            asv_embeddings, counts = inputs
            mask = tf.cast(counts > 0, dtype=tf.float32)

        sample_embeddings = tf.reduce_sum(asv_embeddings, axis=1) / tf.reduce_sum(
            mask, axis=1
        )

        sample_embeddings = self.ff(sample_embeddings, training=True)
        output = self.out_ff(sample_embeddings)
        return sample_embeddings, output

    def get_config(self):
        config = super(SequenceRegressorV2, self).get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_hidden_layers": self.num_hidden_layers,
                "base_output_dim": self.base_output_dim,
                "shift": self.shift,
                "scale": self.scale,
                "dropout_rate": self.dropout_rate,
                "embedding_dim": self.embedding_dim,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "intermediate_activation": self.intermediate_activation,
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

        if self.base_model is not None:
            config.update(
                {"base_model": tf.keras.saving.serialize_keras_object(self.base_model)}
            )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if hasattr(config, "base_model"):
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
