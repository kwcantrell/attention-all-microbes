from __future__ import annotations

import tensorflow as tf

from aam.layers import ASVEncoder


@tf.keras.saving.register_keras_serializable(package="NucleotideEncoder")
class NucleotideEncoder(tf.keras.Model):
    def __init__(
        self,
        embedding_dim: int,
        max_bp: int,
        dropout_rate: float,
        intermediate_activation: str = "gelu",
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 256,
        normalize_outputs: bool = True,
        use_residual_connections: bool = False,
        **kwargs,
    ):
        super(NucleotideEncoder, self).__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.max_bp = max_bp
        self.dropout_rate = dropout_rate
        self.intermediate_activation = intermediate_activation
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.normalize_outputs = normalize_outputs
        self.use_residual_connections = use_residual_connections

        self.loss_tracker = tf.keras.metrics.Mean()
        self.nuc_tracker = tf.keras.metrics.Mean()

    def build(self, input_shape):
        if self.built:
            return

        self.asv_encoder = ASVEncoder(
            self.max_bp,
            self.attention_heads,
            self.attention_layers,
            self.dropout_rate,
            self.intermediate_size,
            intermediate_activation=self.intermediate_activation,
            embedding_dim=self.embedding_dim,
            normalize_outputs=self.normalize_outputs,
            use_residual_connections=self.use_residual_connections,
            name="asv_encoder",
        )
        super(NucleotideEncoder, self).build(input_shape)

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            _, nuc_loss = self(inputs, training=True)
            nuc_loss = tf.reduce_mean(nuc_loss)
            loss = nuc_loss

            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.nuc_tracker.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def test_step(self, inputs):
        _, nuc_loss = self(inputs, training=False)
        nuc_loss = tf.reduce_mean(nuc_loss)
        self.loss_tracker.update_state(nuc_loss)
        self.nuc_tracker.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        tokens = inputs
        embeddings, loss = self.asv_encoder(tokens, include_bert_random_mask=training, training=training)
        return embeddings, loss

    def get_config(self):
        config = super(NucleotideEncoder, self).get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "max_bp": self.max_bp,
                "dropout_rate": self.dropout_rate,
                "intermediate_activation": self.intermediate_activation,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "normalize_outputs": self.normalize_outputs,
                "use_residual_connections": self.use_residual_connections,
            }
        )
        return config
