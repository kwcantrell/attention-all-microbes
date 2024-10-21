from __future__ import annotations

from typing import Optional, Union

import tensorflow as tf
import tensorflow_models as tfm

from aam.unifrac_model import UnifracModel
from aam.utils import apply_random_mask, float_mask, masked_loss


@tf.keras.saving.register_keras_serializable(package="TransferLearnNucleotideModel")
class TransferLearnNucleotideModel(tf.keras.Model):
    def __init__(
        self,
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
        **kwargs,
    ):
        super(TransferLearnNucleotideModel, self).__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.num_classes = num_classes
        self.shift = shift
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.num_tax_levels = num_tax_levels
        self.loss_tracker = tf.keras.metrics.Mean()

        self.embeddings_scale = tf.keras.layers.Dense(embedding_dim, activation="relu")
        self.embeddings_norm = tf.keras.layers.LayerNormalization(epsilon=0.000001)

        # layers used in model
        self.base_model = UnifracModel(
            self.embedding_dim,
            150,
            sample_attention_heads=self.attention_heads,
            sample_attention_layers=self.attention_layers,
            sample_attention_ff=self.intermediate_size,
            dropout_rate=0.0,
            nuc_attention_heads=1,
            nuc_attention_layers=3,
            intermediate_ff=128,
        )

        self.count_encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
        )
        self.count_pos = tfm.nlp.layers.PositionEmbedding(515, dtype=tf.float32)
        self.count_out = tf.keras.layers.Dense(1, dtype=tf.float32)
        self.count_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        self.count_loss = tf.keras.losses.MeanSquaredError(reduction="none")
        self.count_tracker = tf.keras.metrics.Mean()

        self.tax_encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
        )
        self.tax_pos = tfm.nlp.layers.PositionEmbedding(515)

        if self.num_tax_levels is not None:
            self.tax_level_logits = tf.keras.layers.Dense(
                self.num_tax_levels,
                dtype=tf.float32,
            )
            self.tax_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                ignore_class=0, from_logits=True, reduction="none"
            )
        self.tax_tracker = tf.keras.metrics.Mean()

        self.target_encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
        )
        self.target_pos = tfm.nlp.layers.PositionEmbedding(515, dtype=tf.float32)
        self.target_tracker = tf.keras.metrics.Mean()

        if self.num_classes is None:
            self.target_ff = tf.keras.layers.Dense(1, dtype=tf.float32)
            self.metric_tracker = tf.keras.metrics.MeanAbsoluteError()
            # self.metric_tracker = tf.keras.metrics.MeanSquaredError()
            self.metric_string = "mae"
            self.target_activation = tf.keras.layers.Activation(
                "linear", dtype=tf.float32
            )
        else:
            self.target_ff = tf.keras.layers.Dense(num_classes, dtype=tf.float32)
            self.metric_tracker = tf.keras.metrics.CategoricalAccuracy()
            self.metric_string = "accuracy"
            self.target_activation = tf.keras.layers.Activation(
                "softmax", dtype=tf.float32
            )
        self.loss_metrics = sorted(
            ["loss", "target_loss", "count_mse", self.metric_string]
        )

    def evaluate_metric(self, dataset, metric, **kwargs):
        metric_index = self.loss_metrics.index(metric)
        evaluated_metrics = super(TransferLearnNucleotideModel, self).evaluate(
            dataset, **kwargs
        )
        return evaluated_metrics[metric_index]

    def _compute_target_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(self.loss(y_true, y_pred))

    @masked_loss(sparse_cat=False)
    def _compute_count_loss(
        self, counts: tf.Tensor, count_pred: tf.Tensor
    ) -> tf.Tensor:
        relative_counts = self._relative_abundance(counts)
        return self.count_loss(relative_counts, count_pred)

    @masked_loss(sparse_cat=True)
    def _compute_tax_loss(
        self,
        tax_tokens: tf.Tensor,
        tax_pred: tf.Tensor,
        sample_weights: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        token_shape = tf.shape(tax_tokens)
        loss = self.tax_loss(tax_tokens, tax_pred)
        # if sample_weights is not None:
        #     sample_weights = tf.reshape(sample_weights, token_shape)
        #     loss = loss * sample_weights

        return loss

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
        if self.num_tax_levels is None:
            y_target = y_true
            count_pred, y_pred, nuc_pred = outputs
            tax_loss = 0.0
        else:
            y_target, tax_tokens = y_true
            count_pred, y_pred, tax_pred, nuc_pred = outputs
            tax_loss = self._compute_tax_loss(
                tax_tokens, tax_pred, sample_weights=sample_weights
            )

        target_loss = self._compute_target_loss(y_target, y_pred)
        count_loss = self._compute_count_loss(counts, count_pred)
        nuc_loss = self.base_model._compute_nuc_loss(nuc_tokens, nuc_pred)

        loss = target_loss + tax_loss + nuc_loss + count_loss

        return (
            loss,
            target_loss,
            count_loss,
            tax_loss,
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
        if self.num_tax_levels is None:
            _, y_pred, _ = outputs
        else:
            y_true, _ = y_true
            _, y_pred, _, _ = outputs

        if self.num_classes is None:
            y_true = y_true * self.scale + self.shift
            y_pred = y_pred * self.scale + self.shift
            self.metric_tracker(y_true, y_pred)
            # self.metric_tracker(y_true, y_pred)
        else:
            y_true = tf.cast(y_true, dtype=tf.int32)
            y_true = tf.one_hot(y_true, depth=self.num_classes)
            self.metric_tracker(y_true, y_pred)

    def build(self, input_shape=None):
        super(TransferLearnNucleotideModel, self).build(
            [tf.TensorShape([None, None, 150]), tf.TensorShape([None, None, 1])]
        )

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, (y, _) = data
        if self.num_tax_levels is None:
            _, y_pred, _ = self(inputs, training=False)
        else:
            y, _ = y
            _, y_pred, _, _ = self(inputs, training=False)

        if self.num_classes is None:
            y_true = y
            y_true = y_true * self.scale + self.shift
            y_pred = y_pred * self.scale + self.shift
            return y_pred, y_true
        return tf.argmax(y_pred, axis=-1), y

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, (y, sample_weights) = data

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, target_loss, count_mse, tax_loss, nuc_loss = self._compute_loss(
                inputs, y, outputs, sample_weights
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.target_tracker.update_state(target_loss)
        self.count_tracker.update_state(count_mse)
        self.tax_tracker.update_state(tax_loss)
        self.base_model.entropy.update_state(nuc_loss)
        self._compute_metric(y, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "target_loss": self.target_tracker.result(),
            "count_mse": self.count_tracker.result(),
            "tax_entoropy": self.tax_tracker.result(),
            "nuc_entropy": self.base_model.entropy.result(),
            self.metric_string: self.metric_tracker.result(),
            # "mse": self.target_tracker.result(),
        }

    def test_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, (y, sample_weights) = data

        outputs = self(inputs, training=False)
        loss, target_loss, count_mse, tax_loss, nuc_loss = self._compute_loss(
            inputs, y, outputs, sample_weights
        )

        self.loss_tracker.update_state(loss)
        self.target_tracker.update_state(target_loss)
        self.count_tracker.update_state(count_mse)
        self.tax_tracker.update_state(tax_loss)
        self.base_model.entropy.update_state(nuc_loss)
        self._compute_metric(y, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "target_loss": self.target_tracker.result(),
            "count_mse": self.count_tracker.result(),
            "tax_entoropy": self.tax_tracker.result(),
            "nuc_entropy": self.base_model.entropy.result(),
            self.metric_string: self.metric_tracker.result(),
            # "mse": self.target_tracker.result(),
        }

    def _relative_abundance(self, counts: tf.Tensor) -> tf.Tensor:
        counts = tf.cast(counts, dtype=tf.float32)
        count_sums = tf.reduce_sum(counts, axis=1, keepdims=True)
        rel_abundance = counts / count_sums
        return rel_abundance

    def _compute_tax_embeddings(
        self,
        tensor: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        tax_embeddings = tensor + self.tax_pos(tensor)
        tax_embeddings = self.tax_encoder(
            tax_embeddings,
            attention_mask=attention_mask > 0,
            training=training,
        )
        tax_pred = tax_embeddings[:, :-1, :]
        tax_pred = self.tax_level_logits(tax_pred)
        return tax_embeddings, tax_pred

    def _compute_count_embeddings(
        self,
        tensor: tf.Tensor,
        relative_abundances: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        count_embeddings = tensor + self.count_pos(tensor) * relative_abundances
        count_embeddings = self.count_encoder(
            count_embeddings,
            attention_mask=attention_mask > 0,
            training=training,
        )
        count_pred = count_embeddings[:, :-1, :]
        count_pred = self.count_out(count_pred)
        return count_embeddings, count_pred

    def _compute_target_embeddings(
        self,
        tensor: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        target_embeddings = tensor + self.target_pos(tensor)
        target_embeddings = self.target_encoder(
            target_embeddings,
            attention_mask=attention_mask > 0,
            training=training,
        )
        target_out = target_embeddings[:, -1, :]
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
        if training:
            rel_abundance = apply_random_mask(rel_abundance, mask_percent=0.1)

        # account for <SAMPLE> token
        count_mask = tf.pad(count_mask, [[0, 0], [0, 1], [0, 0]], constant_values=1)
        rel_abundance = tf.pad(
            rel_abundance, [[0, 0], [0, 1], [0, 0]], constant_values=1
        )
        count_attention_mask = tf.matmul(count_mask, count_mask, transpose_b=True)

        base_embeddings, nuc_embeddings = self.base_model(
            tokens,
            return_nuc_embeddings=True,
            randomly_mask_nucleotides=True,
            training=training,
        )
        base_embeddings = self.embeddings_scale(base_embeddings)
        base_embeddings = self.embeddings_norm(base_embeddings)

        tax_embeddings, tax_pred = self._compute_tax_embeddings(
            base_embeddings, attention_mask=count_attention_mask, training=training
        )

        count_embeddings, count_pred = self._compute_count_embeddings(
            tax_embeddings,
            rel_abundance,
            attention_mask=count_attention_mask,
            training=training,
        )

        target_embeddings, target_out = self._compute_target_embeddings(
            count_embeddings, attention_mask=count_attention_mask, training=training
        )

        if self.num_tax_levels is None:
            return (
                self.count_activation(count_pred),
                self.target_activation(target_out),
                nuc_embeddings,
            )

        return (
            self.count_activation(count_pred),
            self.target_activation(target_out),
            tax_pred,
            nuc_embeddings,
        )

    def get_config(self):
        config = super(TransferLearnNucleotideModel, self).get_config()
        config.update(
            {
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "shift": self.shift,
                "scale": self.scale,
                "dropout": self.dropout_rate,
                "num_tax_levels": self.num_tax_levels,
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
