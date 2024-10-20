from __future__ import annotations

from typing import Optional, Union

import tensorflow as tf
import tensorflow_models as tfm

from aam.losses import PairwiseLoss
from aam.models.base_sequence_encoder import BaseSequenceEncoder
from aam.utils import float_mask


@tf.keras.saving.register_keras_serializable(package="UniFracEncoder")
class UniFracEncoder(tf.keras.Model):
    def __init__(
        self,
        token_limit: int,
        dropout_rate: float = 0.0,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 1024,
        intermediate_activation: str = "relu",
        **kwargs,
    ):
        super(UniFracEncoder, self).__init__(**kwargs)

        self.token_limit = token_limit
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation

        self.loss_tracker = tf.keras.metrics.Mean()
        self.unifrac_loss = PairwiseLoss()
        self.unifrac_tracker = tf.keras.metrics.Mean()

        # layers used in model
        self.base_encoder = BaseSequenceEncoder(
            self.embedding_dim,
            150,
            self.token_limit,
            sample_attention_heads=self.attention_heads,
            sample_attention_layers=self.attention_layers,
            sample_intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            nuc_attention_heads=1,
            nuc_attention_layers=3,
            nuc_intermediate_size=128,
            intermediate_activation=self.intermediate_activation,
            name="base_encoder",
        )

        self.embeddings_scale = tf.keras.layers.Dense(
            embedding_dim,
            activation=self.intermediate_activation,
            name="embeddings_scale",
        )
        self.embeddings_norm = tf.keras.layers.LayerNormalization(
            epsilon=0.000001, name="embeddings_norm"
        )

        self.unifrac_encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
            name="unifrac_encoder",
        )
        self.unifrac_pos = tfm.nlp.layers.PositionEmbedding(
            self.token_limit, name="unifrac_pos"
        )

        self.unifrac_ff = tf.keras.layers.Dense(
            self.embedding_dim, dtype=tf.float32, name="unifrac_ff"
        )

        self.loss_metrics = sorted(["loss", "target_loss", "count_mse"])

    def evaluate_metric(self, dataset, metric, **kwargs):
        metric_index = self.loss_metrics.index(metric)
        evaluated_metrics = super(UniFracEncoder, self).evaluate(dataset, **kwargs)
        return evaluated_metrics[metric_index]

    def _compute_nuc_loss(self, nuc_tokens, nuc_pred):
        return self.base_encoder._compute_nuc_loss(nuc_tokens, nuc_pred)

    def _compute_unifrac_loss(
        self,
        y_true: tf.Tensor,
        unifrac_embeddings: tf.Tensor,
    ) -> tf.Tensor:
        loss = self.unifrac_loss(y_true, unifrac_embeddings)
        num_samples = tf.reduce_sum(float_mask(loss))
        return tf.math.divide_no_nan(tf.reduce_sum(loss), num_samples)

    def _compute_loss(
        self,
        model_inputs: tuple[tf.Tensor, tf.Tensor],
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        nuc_tokens, counts = model_inputs
        embeddings, unifrac_embeddings, nuc_pred = outputs
        tax_loss = self._compute_unifrac_loss(y_true, unifrac_embeddings)

        nuc_loss = self._compute_nuc_loss(nuc_tokens, nuc_pred)

        loss = tax_loss + nuc_loss

        return [loss, tax_loss, nuc_loss]

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, sample_ids = data
        embeddings, _, _ = self(inputs, training=False)

        return embeddings, sample_ids

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, tax_loss, nuc_loss = self._compute_loss(inputs, y, outputs)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.unifrac_tracker.update_state(tax_loss)
        self.base_encoder.nuc_entropy.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "unifrac_mse": self.unifrac_tracker.result(),
            "nuc_entropy": self.base_encoder.nuc_entropy.result(),
        }

    def test_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data

        outputs = self(inputs, training=False)
        loss, tax_loss, nuc_loss = self._compute_loss(inputs, y, outputs)

        self.loss_tracker.update_state(loss)
        self.unifrac_tracker.update_state(tax_loss)
        self.base_encoder.nuc_entropy.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "unifrac_mse": self.unifrac_tracker.result(),
            "nuc_entropy": self.base_encoder.nuc_entropy.result(),
        }

    def _compute_sequece_embeddings(
        self,
        tensor: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        base_embeddings, nuc_embeddings = self.base_encoder(tensor, training=training)
        base_embeddings = self.embeddings_scale(base_embeddings)
        base_embeddings = self.embeddings_norm(base_embeddings)
        return base_embeddings, nuc_embeddings

    def _compute_embeddings(
        self,
        tensor: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        embeddings = tensor + self.unifrac_pos(tensor)
        embeddings = self.unifrac_encoder(
            embeddings,
            attention_mask=attention_mask > 0,
            training=training,
        )
        unifrac_embedding = embeddings[:, 0, :]
        unifrac_embedding = self.unifrac_ff(unifrac_embedding)
        return embeddings, unifrac_embedding

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.float32)

        count_mask = float_mask(counts, dtype=tf.int32)

        # account for <SAMPLE> token
        count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        count_attention_mask = tf.matmul(count_mask, count_mask, transpose_b=True)

        base_embeddings, nuc_embeddings = self._compute_sequece_embeddings(
            tokens, training=training
        )

        embeddings, unifrac_embedding = self._compute_embeddings(
            base_embeddings, attention_mask=count_attention_mask, training=training
        )

        return [embeddings, unifrac_embedding, nuc_embeddings]

    def get_config(self):
        config = super(UniFracEncoder, self).get_config()
        config.update(
            {
                "token_limit": self.token_limit,
                "dropout_rate": self.dropout_rate,
                "embedding_dim": self.embedding_dim,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "intermediate_activation": self.intermediate_activation,
            }
        )
        return config
