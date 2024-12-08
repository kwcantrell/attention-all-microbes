from __future__ import annotations

import tensorflow as tf

# from aam.models.attention_pooling import AttentionPooling
from aam.layers import (
    ASVEncoder,
)


@tf.keras.saving.register_keras_serializable(package="SequenceEncoder")
class NucleotideEncoder(tf.keras.Model):
    def __init__(
        self,
        embedding_dim: int,
        max_bp: int,
        dropout_rate: float,
        intermediate_activation: str = "gelu",
        add_token: bool = True,
        **kwargs,
    ):
        super(NucleotideEncoder, self).__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.max_bp = max_bp
        self.dropout_rate = dropout_rate
        self.intermediate_activation = intermediate_activation
        self.add_token = add_token

        self.asv_encoder = ASVEncoder(
            self.max_bp,
            4,
            4,
            self.dropout_rate,
            256,
            add_token=self.add_token,
            embedding_dim=self.embedding_dim,
            name="asv_encoder",
        )
        self.nuc_loss = tf.keras.losses.CategoricalCrossentropy(reduction="none")
        self.loss_tracker = tf.keras.metrics.Mean()

    def _compute_loss(
        self,
        model_inputs: tuple[tf.Tensor, tf.Tensor],
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        nuc_tokens, _ = model_inputs
        _, nuc_mask, nuc_pred = outputs

        nuc_tokens = nuc_tokens + self.asv_encoder.nucleotide_position
        nuc_tokens = tf.reshape(nuc_tokens, shape=[-1])
        nuc_mask = tf.reshape(nuc_mask, shape=[-1])
        nuc_tokens = nuc_tokens[nuc_mask]
        nuc_tokens = tf.one_hot(nuc_tokens, tf.shape(nuc_pred)[-1])
        nuc_loss = self.nuc_loss(nuc_tokens, nuc_pred)
        nuc_loss = tf.reduce_mean(nuc_loss)
        loss = nuc_loss
        return loss

    def train_step(self, data):
        inputs, _ = data
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss = self._compute_loss(inputs, outputs)

        gradients = tape.gradient(
            loss,
            self.trainable_variables,
        )
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def test_step(self, data):
        inputs, _ = data
        outputs = self(inputs, training=False)
        loss = self._compute_loss(inputs, outputs)
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        if isinstance(inputs, (list, tuple)):
            tokens, _ = inputs
        else:
            tokens = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)

        asv_embeddings, nuc_mask, nuc_pred = self.asv_encoder(tokens, training=training)

        return asv_embeddings, nuc_mask, nuc_pred

    def get_config(self):
        config = super(NucleotideEncoder, self).get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "max_bp": self.max_bp,
                "dropout_rate": self.dropout_rate,
                "intermediate_activation": self.intermediate_activation,
                "add_token": self.add_token,
            }
        )
        return config
