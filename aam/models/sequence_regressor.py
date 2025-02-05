from __future__ import annotations

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
        self.embedding_loss = PairwiseLoss()

        # self.base_losses = {"base_loss": self.base_model._compute_encoder_loss}
        # self.base_metrics = {"base_loss": ["encoder_loss", self.base_model.encoder_tracker]}

        # if self.freeze_base:
        #     print("Freezing base model...")
        #     self.base_model.trainable = False

        # self.count_encoder = TransformerEncoder(
        #     num_layers=self.attention_layers,
        #     num_attention_heads=self.attention_heads,
        #     intermediate_size=intermediate_size,
        #     dropout_rate=self.dropout_rate,
        #     activation=self.intermediate_activation,
        # )
        # self.count_pos = tfm.nlp.layers.PositionEmbedding(self.token_limit + 5, initializer="zeros")
        # self.count_out = tf.keras.layers.Dense(1, dtype=tf.float32)
        # self._rezero_a = self.add_weight(
        #     name="rezero_alpha",
        #     initializer=tf.keras.initializers.Zeros(),
        #     trainable=True,
        #     dtype=tf.float32,
        # )
        self.count_loss = tf.keras.losses.MeanSquaredError(reduction="none")
        self.embedding_tracker = tf.keras.metrics.Mean()

        # self.target_encoder = TransformerEncoder(
        #     num_layers=self.attention_layers,
        #     num_attention_heads=self.attention_heads,
        #     intermediate_size=intermediate_size,
        #     dropout_rate=self.dropout_rate,
        #     activation=self.intermediate_activation,
        # )

        self.target_tracker = tf.keras.metrics.Mean()
        if not self.classifier:
            self.metric_tracker = tf.keras.metrics.MeanAbsoluteError()
            self.metric_string = "mae"
        else:
            self.metric_tracker = tf.keras.metrics.SparseCategoricalAccuracy()
            self.metric_string = "accuracy"

        # self.attention_pooling = MultiHeadAttentionPooling()
        # self.target_ff = tf.keras.layers.Dense(self.out_dim, dtype=tf.float32)

        self.loss_metrics = sorted(["loss", "target_loss", "count_mse", self.metric_string])
        self.gradient_accumulator = GradientAccumulator(self.accumulation_steps)
        self.loss_scaler = LossScaler(self.gradient_accumulator.accum_steps)

    def build(self, input_shape):
        if self.built:
            return

        if self.freeze_base:
            print("Freezing base model...")
            self.base_model.trainable = False

        self.encoder = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
            normalize_outputs=self.normalize_outputs,
            use_residual_connections=self.use_residual_connections,
            use_linear_bias=self.use_linear_bias,
            name="encoder",
        )

        self.attention_pooling = MultiHeadAttentionPooling(
            self.normalize_outputs,
            num_heads=self.attention_heads,
            use_residual_connections=self.use_residual_connections,
            use_linear_bias=self.use_linear_bias,
        )

        self._rezero = self.add_weight(
            name="rezero_alpha", initializer=tf.keras.initializers.Zeros(), trainable=True, dtype=tf.float32
        )
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(
            2048, seq_axis=1, dtype=tf.float32
        )

        self.output_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        self.target_ff = tf.keras.layers.Dense(self.out_dim, dtype=tf.float32)
        self.sample_ff = tf.keras.layers.Dense(self.embedding_dim, dtype=tf.float32)
        super(SequenceRegressor, self).build(input_shape)

    def evaluate_metric(self, dataset, metric, **kwargs):
        metric_index = self.loss_metrics.index(metric)
        evaluated_metrics = super(SequenceRegressor, self).evaluate(dataset, **kwargs)
        return evaluated_metrics[metric_index]

    def _compute_target_loss(self, y_true: tf.Tensor, model_outputs: tf.Tensor) -> tf.Tensor:
        embeddings, y_pred = model_outputs

        # step 1: pairwise distance of embeddings should match pairwise distance of target
        y_true_dist = _pairwise_distances(tf.reshape(y_true, shape=[-1, 1]), squared=False)
        embedding_loss = self.embedding_loss(y_true_dist, embeddings)

        # step 2: minimize mse
        mse_loss = tf.square(y_true - y_pred)
        return mse_loss, embedding_loss

    def _compute_count_loss(self, counts: tf.Tensor, count_pred: tf.Tensor, count_mask) -> tf.Tensor:
        count_mask = counts > 0
        count_mask = tf.reshape(count_mask, shape=[-1])
        relative_counts = tf.reshape(self._relative_abundance(counts), shape=[-1])[count_mask]
        count_pred = tf.reshape(count_pred, shape=[-1])[count_mask]

        loss = tf.square(tf.math.log(relative_counts) - tf.math.log(count_pred))
        loss = tf.reduce_mean(loss)
        return loss

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

        # step 1: minimize mse
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # step 2: pairwise distance of sample_embeddings should match pairwise distance of target
        # y_true_dist = _pairwise_distances(y_true, squared=False)
        # embedding_loss = tf.reduce_mean(self.embedding_loss(y_true_dist, sample_embeddings))
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
        inputs, y_true = data
        target_embeddings, count_pred, y_pred, base_pred, nuc_mask, nuc_pred = self(inputs, training=False)

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

        # shape = tf.shape(encoder_target)
        # group_dim = shape[-1]
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

    def _extract_asv_embeddings(self, inputs):
        asv_embeddings, counts = self.base_model(inputs, return_asv_embeddings=True, training=False)
        return asv_embeddings, counts

    def _group_embeddings(self, asv_embeddings, inputs, group, samples_per_group):
        base_inputs = self.base_model._group_embeddings(asv_embeddings, inputs, group, samples_per_group)
        return self.base_model(base_inputs, return_asv_embeddings=True, training=False)

    def call(
        self, inputs, training: bool = False
    ) -> Union[
        tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        training = training and self.trainable

        # get denoised embeddings and attention mask
        asv_embeddings, counts = self.base_model(inputs, return_asv_embeddings=True, training=False)
        mask = tf.cast(counts > 0, dtype=self.compute_dtype)

        # compute relative abundance
        counts = tf.cast(counts, dtype=tf.float32)
        total_counts = tf.reduce_sum(counts, axis=1, keepdims=True)
        counts = counts / total_counts
        counts = counts * tf.cast(self._rezero, dtype=tf.float32) * tf.cast(self.pos_emb(counts), dtype=tf.float32)

        # compute sample embeddings and target
        asv_embeddings = tf.cast(asv_embeddings, dtype=tf.float32) +  counts
        asv_embeddings = self.encoder(tf.cast(asv_embeddings, dtype=self.compute_dtype), mask=mask, training=training)
        sample_embedding = self.attention_pooling(asv_embeddings, mask=mask, training=training)
        return self.sample_ff(sample_embedding), self.output_activation(self.target_ff(sample_embedding))

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
        config["base_model"] = tf.keras.saving.deserialize_keras_object(config["base_model"])
        model = cls(**config)

        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        if input_shape is not None:
            model.build(input_shape)

        return model
