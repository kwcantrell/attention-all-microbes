from __future__ import annotations

from typing import Optional, Union

import tensorflow as tf
import tensorflow_models as tfm

from aam.models.base_sequence_encoder import BaseSequenceEncoder
from aam.models.transformers import TransformerEncoder
from aam.utils import float_mask, masked_loss


@tf.keras.saving.register_keras_serializable(package="TaxonomyEncoder")
class TaxonomyEncoder(tf.keras.Model):
    def __init__(
        self,
        num_tax_levels: int,
        token_limit: int,
        dropout_rate: float = 0.0,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 1024,
        intermediate_activation: str = "relu",
        **kwargs,
    ):
        super(TaxonomyEncoder, self).__init__(**kwargs)

        self.num_tax_levels = num_tax_levels
        self.token_limit = token_limit
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation

        self.loss_tracker = tf.keras.metrics.Mean()
        self.tax_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            ignore_class=0, from_logits=True, reduction="none"
        )
        self.tax_tracker = tf.keras.metrics.Mean()

        # layers used in model
        self.base_encoder = BaseSequenceEncoder(
            self.embedding_dim,
            150,
            self.token_limit,
            sample_attention_heads=self.attention_heads,
            sample_attention_layers=self.attention_layers,
            sample_intermediate_size=self.intermediate_size,
            dropout_rate=0.0,
            nuc_attention_heads=1,
            nuc_attention_layers=3,
            nuc_intermediate_size=128,
            intermediate_activation=self.intermediate_activation,
            name="base_encoder",
        )

        self._tax_alpha = self.add_weight(
            name="tax_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )
        self.embeddings_scale = tf.keras.layers.Dense(
            embedding_dim,
            activation=self.intermediate_activation,
            name="embeddings_scale",
        )
        self.embeddings_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.tax_encoder = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
            name="tax_encoder",
        )
        self.tax_pos = tfm.nlp.layers.PositionEmbedding(
            self.token_limit, name="tax_pos"
        )

        self.tax_level_logits = tf.keras.layers.Dense(
            self.num_tax_levels, dtype=tf.float32, name="tax_level_logits"
        )

        self.loss_metrics = sorted(["loss", "target_loss", "count_mse"])

    def evaluate_metric(self, dataset, metric, **kwargs):
        metric_index = self.loss_metrics.index(metric)
        evaluated_metrics = super(TaxonomyEncoder, self).evaluate(dataset, **kwargs)
        return evaluated_metrics[metric_index]

    def _compute_nuc_loss(self, nuc_tokens, nuc_pred):
        return self.base_encoder._compute_nuc_loss(nuc_tokens, nuc_pred)

    @masked_loss(sparse_cat=True)
    def _compute_tax_loss(
        self,
        tax_tokens: tf.Tensor,
        tax_pred: tf.Tensor,
        sample_weights: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        loss = self.tax_loss(tax_tokens, tax_pred)

        return loss

    def _compute_loss(
        self,
        model_inputs: tuple[tf.Tensor, tf.Tensor],
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        sample_weights: Optional[tf.Tensor] = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        nuc_tokens, counts = model_inputs
        taxonomy_embeddings, tax_pred, nuc_pred = outputs
        tax_loss = self._compute_tax_loss(
            y_true, tax_pred, sample_weights=sample_weights
        )

        nuc_loss = self._compute_nuc_loss(nuc_tokens, nuc_pred)

        loss = tax_loss + nuc_loss

        return (loss, tax_loss, nuc_loss)

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, (y, _) = data
        embeddings, _, _ = self(inputs, training=False)

        return embeddings, y

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
        self.tax_tracker.update_state(tax_loss)
        self.base_encoder.nuc_entropy.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "tax_entropy": self.tax_tracker.result(),
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
        self.tax_tracker.update_state(tax_loss)
        self.base_encoder.nuc_entropy.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "tax_entropy": self.tax_tracker.result(),
            "nuc_entropy": self.base_encoder.nuc_entropy.result(),
        }

    def _compute_sequece_embeddings(
        self,
        tensor: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        base_embeddings, nuc_embeddings = self.base_encoder(tensor, training=training)
        base_embeddings = self.embeddings_scale(base_embeddings)
        base_embeddings = self.embeddings_norm(base_embeddings)
        return base_embeddings, nuc_embeddings

    def _compute_tax_embeddings(
        self,
        tensor: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        tax_embeddings = tensor + self.tax_pos(tensor)
        tax_embeddings = self.tax_encoder(
            tax_embeddings, mask=attention_mask, training=training
        )
        tax_pred = tax_embeddings[:, 1:, :]
        tax_pred = self.tax_level_logits(tax_pred)
        return tax_embeddings, tax_pred

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        # base_embeddings, nuc_embeddings = self._compute_sequece_embeddings(
        #     tokens, training=training
        # )
        sample_embeddings, nuc_embeddings = self.base_encoder(tokens, training=training)

        # account for <SAMPLE> token
        count_mask = float_mask(counts, dtype=tf.int32)
        count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        count_attention_mask = count_mask

        tax_gated_embeddings, tax_pred = self._compute_tax_embeddings(
            sample_embeddings, attention_mask=count_attention_mask, training=training
        )

        tax_embeddings = sample_embeddings + tax_gated_embeddings * self._tax_alpha

        return [tax_embeddings, tax_pred, nuc_embeddings]

    def get_config(self):
        config = super(TaxonomyEncoder, self).get_config()
        config.update(
            {
                "num_tax_levels": self.num_tax_levels,
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
