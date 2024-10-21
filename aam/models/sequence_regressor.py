from __future__ import annotations

from typing import Optional, Union

import tensorflow as tf
import tensorflow_models as tfm

from aam.models.taxonomy_encoder import TaxonomyEncoder
from aam.models.transformers import TransformerEncoder
from aam.models.unifrac_encoder import UniFracEncoder
from aam.utils import float_mask, masked_loss


@tf.keras.saving.register_keras_serializable(package="SequenceRegressor")
class SequenceRegressor(tf.keras.Model):
    def __init__(
        self,
        token_limit: int,
        num_classes: Optional[int] = None,
        shift: float = 0.0,
        scale: float = 1.0,
        dropout_rate: float = 0.0,
        num_tax_levels: Optional[int] = None,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 1024,
        intermediate_activation: str = "relu",
        base_model: Union[str, TaxonomyEncoder, UniFracEncoder] = "taxonomy",
        freeze_base: bool = False,
        penalty: float = 1.0,
        **kwargs,
    ):
        super(SequenceRegressor, self).__init__(**kwargs)
        self.token_limit = token_limit
        self.num_classes = num_classes
        self.shift = shift
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.num_tax_levels = num_tax_levels
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.freeze_base = freeze_base
        self.penalty = penalty
        self.loss_tracker = tf.keras.metrics.Mean()

        # layers used in model
        if isinstance(base_model, str):
            if base_model == "taxonomy":
                self.base_model = TaxonomyEncoder(
                    num_tax_levels=self.num_tax_levels,
                    token_limit=self.token_limit,
                    dropout_rate=self.dropout_rate,
                    embedding_dim=self.embedding_dim,
                    attention_heads=self.attention_heads,
                    attention_layers=self.attention_layers,
                    intermediate_size=self.intermediate_size,
                    intermediate_activation=self.intermediate_activation,
                )
            elif base_model == "unifrac":
                self.base_model = UniFracEncoder(
                    self.token_limit,
                    dropout_rate=self.dropout_rate,
                    embedding_dim=self.embedding_dim,
                    attention_heads=self.attention_heads,
                    attention_layers=self.attention_layers,
                    intermediate_size=self.intermediate_size,
                    intermediate_activation=self.intermediate_activation,
                )
            else:
                raise Exception("Invalid base model option.")
        else:
            if not isinstance(base_model, (TaxonomyEncoder, UniFracEncoder)):
                raise Exception(f"Unsupported base model of type {type(base_model)}")
            self.base_model = base_model

        if isinstance(self.base_model, TaxonomyEncoder):
            self.base_losses = {"base_loss": self.base_model._compute_tax_loss}
            self.base_metrics = {
                "base_loss": ("tax_entropy", self.base_model.tax_tracker)
            }
        else:
            self.base_losses = {"base_loss": self.base_model._compute_unifrac_loss}
            self.base_metrics = {
                "base_loss": ["unifrac_mse", self.base_model.unifrac_tracker]
            }

        self.base_losses.update({"nuc_entropy": self.base_model._compute_nuc_loss})
        self.base_metrics.update(
            {"nuc_entropy": ["nuc_entropy", self.base_model.base_encoder.nuc_entropy]}
        )

        if self.freeze_base:
            print("Freezing base model...")
            self.base_model.trainable = False

        self._count_alpha = self.add_weight(
            name="count_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )
        self.count_encoder = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
        )
        self.count_pos = tfm.nlp.layers.PositionEmbedding(
            self.token_limit + 5, dtype=tf.float32
        )
        self.count_out = tf.keras.layers.Dense(1, use_bias=False, dtype=tf.float32)
        self.count_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        self.count_loss = tf.keras.losses.MeanSquaredError(reduction="none")
        self.count_tracker = tf.keras.metrics.Mean()

        self.target_encoder = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
        )
        self.target_tracker = tf.keras.metrics.Mean()

        self.target_ff = tf.keras.layers.Dense(1, use_bias=False, dtype=tf.float32)
        self.metric_tracker = tf.keras.metrics.MeanAbsoluteError()
        self.metric_string = "mae"
        self.target_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        self.loss_metrics = sorted(
            ["loss", "target_loss", "count_mse", self.metric_string]
        )

    def evaluate_metric(self, dataset, metric, **kwargs):
        metric_index = self.loss_metrics.index(metric)
        evaluated_metrics = super(SequenceRegressor, self).evaluate(dataset, **kwargs)
        return evaluated_metrics[metric_index]

    def _compute_target_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(self.loss(y_true, y_pred))

    @masked_loss(sparse_cat=False)
    def _compute_count_loss(
        self, counts: tf.Tensor, count_pred: tf.Tensor
    ) -> tf.Tensor:
        relative_counts = self._relative_abundance(counts)
        loss = tf.square(relative_counts - count_pred)
        return tf.squeeze(loss, axis=-1)

    def _compute_loss(
        self,
        model_inputs: tuple[tf.Tensor, tf.Tensor],
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: Union[
            tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        ],
        sample_weights: Optional[tf.Tensor] = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        nuc_tokens, counts = model_inputs
        y_target, base_target = y_true

        target_embeddings, count_pred, y_pred, base_pred, nuc_pred = outputs

        target_loss = self._compute_target_loss(y_target, y_pred)
        count_loss = self._compute_count_loss(counts, count_pred)

        loss = target_loss + count_loss
        if not self.freeze_base:
            base_loss = (
                self.base_losses["base_loss"](base_target, base_pred) * self.penalty
            )
            nuc_loss = self.base_losses["nuc_entropy"](nuc_tokens, nuc_pred)
            loss = loss + nuc_loss + base_loss
        else:
            base_loss = 0
            nuc_loss = 0

        return (
            loss,
            target_loss,
            count_loss,
            base_loss,
            nuc_loss,
        )

    def _compute_metric(
        self,
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: Union[
            tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        ],
    ):
        y_true, base_target = y_true

        target_embeddings, count_pred, y_pred, base_pred, nuc_pred = outputs
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
        inputs, (y_true, _) = data
        target_embeddings, count_pred, y_pred, base_pred, nuc_pred = self(
            inputs, training=False
        )

        y_true = y_true * self.scale + self.shift
        y_pred = y_pred * self.scale + self.shift
        return y_pred, y_true

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
            loss, target_loss, count_mse, base_loss, nuc_loss = self._compute_loss(
                inputs, y, outputs
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.target_tracker.update_state(target_loss)
        self.count_tracker.update_state(count_mse)
        base_loss_key, base_loss_metric = self.base_metrics["base_loss"]
        base_loss_metric.update_state(base_loss)
        nuc_entropy_key, nuc_entropy_metric = self.base_metrics["nuc_entropy"]
        nuc_entropy_metric.update_state(nuc_loss)
        self._compute_metric(y, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "target_loss": self.target_tracker.result(),
            "count_mse": self.count_tracker.result(),
            base_loss_key: base_loss_metric.result(),
            nuc_entropy_key: nuc_entropy_metric.result(),
            self.metric_string: self.metric_tracker.result(),
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
        loss, target_loss, count_mse, base_loss, nuc_loss = self._compute_loss(
            inputs, y, outputs
        )

        self.loss_tracker.update_state(loss)
        self.target_tracker.update_state(target_loss)
        self.count_tracker.update_state(count_mse)
        base_loss_key, base_loss_metric = self.base_metrics["base_loss"]
        base_loss_metric.update_state(base_loss)
        nuc_entropy_key, nuc_entropy_metric = self.base_metrics["nuc_entropy"]
        nuc_entropy_metric.update_state(nuc_loss)
        self._compute_metric(y, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "target_loss": self.target_tracker.result(),
            "count_mse": self.count_tracker.result(),
            base_loss_key: base_loss_metric.result(),
            nuc_entropy_key: nuc_entropy_metric.result(),
            self.metric_string: self.metric_tracker.result(),
        }

    def _relative_abundance(self, counts: tf.Tensor) -> tf.Tensor:
        counts = tf.cast(counts, dtype=tf.float32)
        count_sums = tf.reduce_sum(counts, axis=1, keepdims=True)
        rel_abundance = counts / count_sums
        return rel_abundance

    def _compute_count_embeddings(
        self,
        tensor: tf.Tensor,
        relative_abundances: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        count_embeddings = tensor + self.count_pos(tensor) * relative_abundances
        count_embeddings = self.count_encoder(
            count_embeddings, mask=attention_mask, training=training
        )
        count_pred = count_embeddings[:, 1:, :]
        count_pred = self.count_out(count_pred)
        return count_embeddings, count_pred

    def _compute_target_embeddings(
        self,
        tensor: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        target_embeddings = self.target_encoder(
            tensor, mask=attention_mask, training=training
        )
        target_out = target_embeddings[:, 0, :]
        target_out = self.target_ff(target_out)
        return target_embeddings, target_out

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> Union[
        tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        count_mask = float_mask(counts, dtype=tf.int32)
        rel_abundance = self._relative_abundance(counts)

        # account for <SAMPLE> token
        count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        rel_abundance = tf.pad(
            rel_abundance, [[0, 0], [1, 0], [0, 0]], constant_values=1
        )
        count_attention_mask = count_mask
        base_embeddings, base_pred, nuc_embeddings = self.base_model(
            (tokens, counts), training=training
        )

        count_gated_embeddings, count_pred = self._compute_count_embeddings(
            base_embeddings,
            rel_abundance,
            attention_mask=count_attention_mask,
            training=training,
        )
        count_embeddings = base_embeddings + count_gated_embeddings * self._count_alpha

        target_embeddings, target_out = self._compute_target_embeddings(
            count_embeddings, attention_mask=count_attention_mask, training=training
        )

        return (
            target_embeddings,
            self.count_activation(count_pred),
            self.target_activation(target_out),
            base_pred,
            nuc_embeddings,
        )

    def get_config(self):
        config = super(SequenceRegressor, self).get_config()
        config.update(
            {
                "token_limit": self.token_limit,
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "num_classes": self.num_classes,
                "shift": self.shift,
                "scale": self.scale,
                "dropout_rate": self.dropout_rate,
                "num_tax_levels": self.num_tax_levels,
                "embedding_dim": self.embedding_dim,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "intermediate_activation": self.intermediate_activation,
                "freeze_base": self.freeze_base,
                "penalty": self.penalty,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["base_model"] = tf.keras.saving.deserialize_keras_object(
            config["base_model"]
        )
        model = cls(**config)
        return model
