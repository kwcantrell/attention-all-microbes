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
        mask_percent: int = 25,
        num_classes: Optional[int] = None,
        shift: float = 0.0,
        scale: float = 1.0,
        penalty: int = 5000,
        dropout: float = 0.0,
        num_tax_levels: Optional[int] = None,
        **kwargs,
    ):
        super(TransferLearnNucleotideModel, self).__init__(**kwargs)

        self.token_dim = 256
        self.mask_percent = mask_percent
        self.num_classes = num_classes
        self.shift = shift
        self.scale = scale
        self.penalty = penalty
        self.dropout = dropout
        self.num_tax_levels = num_tax_levels
        self.loss_tracker = tf.keras.metrics.Mean()
        self.target_tracker = tf.keras.metrics.Mean()

        # layers used in model
        self.base_model = UnifracModel(
            self.token_dim,
            150,
            attention_heads=2,
            attention_layers=2,
            attention_ff=64,
            dropout_rate=0.1,
            penalty=1,
            nuc_attention_heads=1,
            nuc_attention_layers=3,
            intermediate_ff=128,
        )
        self.count_encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=2048,
            dropout_rate=self.dropout,
            activation="relu",
            dtype=tf.float32,
        )
        self.count_out = tf.keras.layers.Dense(1, use_bias=False, dtype=tf.float32)
        self.pos_embeddings = tfm.nlp.layers.PositionEmbedding(515, dtype=tf.float32)
        self.count_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        self.count_loss = tf.keras.losses.MeanSquaredError(reduction="none")
        self.count_tracker = tf.keras.metrics.Mean()

        self.tax_encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=2048,
            dropout_rate=self.dropout,
            activation="relu",
        )

        self.transfer_encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=2048,
            dropout_rate=self.dropout,
            activation="relu",
        )

        if self.num_tax_levels is not None:
            self.tax_level_logits = tf.keras.layers.Dense(
                self.num_tax_levels,
                use_bias=False,
                dtype=tf.float32,
            )
            self.tax_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                ignore_class=0, from_logits=True, reduction="none"
            )
        self.tax_tracker = tf.keras.metrics.Mean()

        if self.num_classes is None:
            self.transfer_ff = tf.keras.layers.Dense(
                1, use_bias=False, dtype=tf.float32
            )
            self.transfer_tracker = tf.keras.metrics.MeanAbsoluteError()
            self.transfer2_tracker = tf.keras.metrics.MeanSquaredError()
            self.transfer_string = "mae"
            self.transfer_activation = tf.keras.layers.Activation(
                "linear", dtype=tf.float32
            )
        else:
            self.transfer_ff = tf.keras.layers.Dense(
                num_classes, use_bias=False, dtype=tf.float32
            )
            self.transfer_tracker = tf.keras.metrics.CategoricalAccuracy()
            self.transfer_string = "accuracy"
            self.transfer_activation = tf.keras.layers.Activation(
                "softmax", dtype=tf.float32
            )
        self.loss_metrics = sorted(
            ["loss", "target_loss", "count_mse", self.transfer_string]
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
        self, tax_tokens: tf.Tensor, tax_pred: tf.Tensor
    ) -> tf.Tensor:
        return self.tax_loss(tax_tokens, tax_pred)

    def _compute_loss(
        self,
        model_inputs: tuple[tf.Tensor, tf.Tensor],
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: Union[
            tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        ],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        nuc_tokens, counts = model_inputs
        if self.num_tax_levels is None:
            y_target = y_true
            count_pred, y_pred, nuc_pred = outputs
            tax_loss = 0.0
        else:
            y_target, tax_tokens = y_true
            count_pred, y_pred, tax_pred, nuc_pred = outputs
            tax_loss = self._compute_tax_loss(tax_tokens, tax_pred)

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
            self.transfer_tracker(y_true, y_pred)
            self.transfer2_tracker(y_true, y_pred)
        else:
            y_true = tf.cast(y_true, dtype=tf.int32)
            y_true = tf.one_hot(y_true, depth=self.num_classes)
            self.transfer_tracker(y_true, y_pred)

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
        if self.num_tax_levels is None:
            inputs, y = data
            _, y_pred, _ = self(inputs, training=False)
        else:
            inputs, y_comb = data
            y, _ = y_comb
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
        inputs, y = data
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, target_loss, count_mse, tax_loss, nuc_loss = self._compute_loss(
                inputs, y, outputs
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
            self.transfer_string: self.transfer_tracker.result(),
            "mse": self.transfer2_tracker.result(),
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
        loss, target_loss, count_mse, tax_loss, nuc_loss = self._compute_loss(
            inputs, y, outputs
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
            self.transfer_string: self.transfer_tracker.result(),
            "mse": self.transfer2_tracker.result(),
        }

    def _relative_abundance(self, counts: tf.Tensor) -> tf.Tensor:
        counts = tf.cast(counts, dtype=tf.float32)
        count_sums = tf.reduce_sum(counts, axis=1, keepdims=True)
        rel_abundance = counts / count_sums
        return rel_abundance

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

        # extended_counts = tf.expand_dims(counts, axis=-1)
        count_mask = float_mask(counts, dtype=tf.int32)
        rel_abundance = self._relative_abundance(counts)

        if training:
            rel_abundance = apply_random_mask(rel_abundance, 0.1)

        # account for <SAMPLE> token
        rel_abundance = tf.pad(
            rel_abundance, [[0, 0], [0, 1], [0, 0]], constant_values=1
        )

        count_embeddings, nuc_embeddings = self.base_model(
            tokens,
            return_nuc_embeddings=True,
            randomly_mask_nucleotides=True,
            training=training,
        )
        count_embeddings = (
            count_embeddings + self.pos_embeddings(count_embeddings) * rel_abundance
        )

        count_mask = tf.pad(count_mask, [[0, 0], [0, 1], [0, 0]], constant_values=1)
        count_attention_mask = tf.matmul(count_mask, count_mask, transpose_b=True)
        count_embeddings = self.count_encoder(
            count_embeddings,
            attention_mask=count_attention_mask > 0,
            training=training,
        )
        count_pred = count_embeddings[:, :-1, :]
        count_pred = self.count_out(count_pred)

        tax_embeddings = self.tax_encoder(
            count_embeddings,
            attention_mask=count_attention_mask > 0,
            training=training,
        )
        tax_pred = tax_embeddings[:, :-1, :]
        tax_pred = self.tax_level_logits(tax_pred)

        target_embeddings = self.transfer_encoder(
            tax_embeddings,
            attention_mask=count_attention_mask > 0,
            training=training,
        )
        target_out = target_embeddings[:, -1, :]
        target_out = self.transfer_ff(target_out)

        if self.num_tax_levels is None:
            return (
                self.count_activation(count_pred),
                self.transfer_activation(target_out),
                nuc_embeddings,
            )

        return (
            self.count_activation(count_pred),
            self.transfer_activation(target_out),
            tax_pred,
            nuc_embeddings,
        )

    def get_config(self):
        config = super(TransferLearnNucleotideModel, self).get_config()
        config.update(
            {
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "mask_percent": self.mask_percent,
                "shift": self.shift,
                "scale": self.scale,
                "penalty": self.penalty,
                "dropout": self.dropout,
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
