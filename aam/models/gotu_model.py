from __future__ import annotations

import inspect

import tensorflow as tf
import tensorflow_models as tfm

from aam.losses import PairwiseLoss
from aam.models.transformer_decoder import TransformerDecoder
from aam.models.unifrac_encoder import UnifracEncoder
from aam.models.utils import sort_using_counts, to_batch


@tf.keras.saving.register_keras_serializable(package="GOTUModel")
class GOTUModel(tf.keras.Model):
    def __init__(
        self,
        dropout_rate: float = 0.0,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 512,
        intermediate_activation: str = "gelu",
        pairwise_loss_type="mse",
        gotu_count=None,
        base_model=None,
        **kwargs,
    ):
        super(GOTUModel, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.pairwise_loss_type = pairwise_loss_type
        self.gotu_count = gotu_count
        self.base_model = base_model
        self.base_model.trainable = False

        self.encoder_tracker = tf.keras.metrics.Mean()
        self.loss_tracker = tf.keras.metrics.Mean()
        self.gotu_tracker = tf.keras.metrics.Mean()

        self.encoder_loss = PairwiseLoss(self.pairwise_loss_type, reduction="none")
        self.gotu_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")

        self.gotu_embedding_layer = tf.keras.layers.Embedding(
            self.gotu_count, self.embedding_dim
        )

        self.gotu_decoder = TransformerDecoder(
            num_attention_heads=self.attention_heads,
            num_layers=self.attention_layers,
            intermediate_size=self.intermediate_size,
            activation=self.intermediate_activation,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.dropout_rate,
            use_linear_bias=True,
        )
        self._softmax = tf.keras.layers.Activation("softmax", dtype=tf.float32)
        self.gotu_output = tf.keras.layers.Dense(self.gotu_count)

    def get_config(self):
        config = super(GOTUModel, self).get_config()
        config.update(
            {
                "dropout_rate": self.dropout_rate,
                "embedding_dim": self.embedding_dim,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "intermediate_activation": self.intermediate_activation,
                "gotu_count": self.gotu_count,
                "base_model": self.base_model.get_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        asv_encoder_config = config.pop("base_model")
        base_model = UnifracEncoder.from_config(asv_encoder_config)
        model = cls(base_model=base_model, **config)
        return model

    def _compute_loss(self, data, outputs):
        inputs, targets = data
        asv_inputs, gotu_inputs = inputs
        asv_targets, gotu_targets = targets
        (gotu_tokens, gotu_batch_indices, gotu_indicies, gotu_counts) = gotu_inputs
        gotu_tokens = tf.expand_dims(gotu_tokens, axis=-1)
        gotu_tokens, gotu_counts = self.base_model.batch_embeddings(
            gotu_tokens, gotu_batch_indices, gotu_counts, gotu_indicies
        )
        gotu_tokens, gotu_counts = sort_using_counts(gotu_tokens, gotu_counts)
        gotu_counts = tf.pad(
            gotu_counts, paddings=[[0, 0], [1, 0], [0, 0]], constant_values=1
        )
        gotu_valid_tokens = tf.cast(tf.squeeze(gotu_counts, axis=-1) > 0, tf.float32)

        gotu_tokens = tf.pad(
            gotu_tokens, paddings=[[0, 0], [0, 1], [0, 0]], constant_values=2
        )
        gotu_pad_mask = tf.cast(gotu_tokens == 0, dtype=tf.int32) * 2
        gotu_tokens = gotu_tokens + gotu_pad_mask

        gotu_tokens = tf.squeeze(gotu_tokens, axis=-1)
        gotu_loss = self.gotu_loss(gotu_tokens, outputs)

        gotu_loss = gotu_loss * gotu_valid_tokens
        gotu_loss = tf.reduce_sum(gotu_loss, axis=-1, keepdims=True) / tf.reduce_sum(
            gotu_valid_tokens, axis=-1, keepdims=True
        )
        gotu_loss = tf.reduce_mean(gotu_loss)
        loss = gotu_loss
        nuc_loss = 0
        unifrac_loss = 0
        return loss, gotu_loss

    def batch_embeddings(self, embeddings, batch_indicies, counts, indices=None):
        emb_dim = tf.shape(embeddings)[-1]
        if indices is not None:
            embeddings = tf.gather(embeddings, indices)
        batch_shape = tf.reduce_max(batch_indicies[:, 0]) + 1
        max_unique = tf.reduce_max(batch_indicies[:, 1]) + 1
        batch_embeddings = tf.scatter_nd(
            batch_indicies, embeddings, shape=[batch_shape, max_unique, emb_dim]
        )
        counts = tf.scatter_nd(
            batch_indicies, counts, shape=[batch_shape, max_unique, 1]
        )
        batch_embeddings, counts = sort_using_counts(batch_embeddings, counts)
        return batch_embeddings, counts

    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, gotu_loss = self._compute_loss(data, outputs)
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.gotu_tracker.update_state(gotu_loss)
        metrics = {
            "loss": self.loss_tracker.result(),
            "gotu_loss": self.gotu_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }
        return metrics

    def test_step(self, data):
        inputs, targets = data
        outputs = self(inputs, training=False)
        loss, gotu_loss = self._compute_loss(data, outputs)

        self.loss_tracker.update_state(loss)
        self.gotu_tracker.update_state(gotu_loss)
        metrics = {
            "loss": self.loss_tracker.result(),
            "gotu_loss": self.gotu_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

        return metrics

    def predict_step(self, data):
        asv_inputs, gotu_inputs = data
        asv_x, _ = asv_inputs
        gotu_x, _ = gotu_inputs
        outputs = self((asv_x, gotu_x), training=True)

        return outputs[0]

    def call(
        self,
        inputs,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        asv_inputs, gotu_inputs = inputs
        (gotu_tokens, gotu_batch_indices, gotu_indicies, gotu_counts) = gotu_inputs
        asv_embeddings, asv_counts = self.base_model.extract_asv_embeddings(
            asv_inputs, batch_embeddings=True, sort_counts=True
        )
        asv_mask = tf.cast(asv_counts > 0, dtype=self.compute_dtype)

        gotu_tokens = tf.expand_dims(gotu_tokens, axis=-1)
        gotu_tokens, gotu_counts = self.base_model.batch_embeddings(
            gotu_tokens, gotu_batch_indices, gotu_counts, gotu_indicies
        )
        gotu_tokens, gotu_counts = sort_using_counts(gotu_tokens, gotu_counts)
        gotu_tokens = tf.pad(
            gotu_tokens, paddings=[[0, 0], [1, 0], [0, 0]], constant_values=1
        )
        gotu_counts = tf.pad(
            gotu_counts, paddings=[[0, 0], [1, 0], [0, 0]], constant_values=1
        )

        gotu_tokens = tf.squeeze(gotu_tokens, axis=-1)
        gotu_pad_mask = tf.cast(gotu_tokens == 0, dtype=tf.int32) * 2
        gotu_tokens = gotu_tokens + gotu_pad_mask
        gotu_embeddings = self.gotu_embedding_layer(gotu_tokens)

        gotu_mask = tf.cast(gotu_counts > 0, dtype=self.compute_dtype)
        gotu_pred = self.gotu_decoder(
            (asv_embeddings, gotu_embeddings), asv_mask, gotu_mask, training=training
        )
        gotu_pred = self.gotu_output(gotu_pred)
        gotu_pred = self._softmax(gotu_pred)

        return gotu_pred
