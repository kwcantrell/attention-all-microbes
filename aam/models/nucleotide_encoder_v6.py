from __future__ import annotations

import tensorflow as tf

from aam.layers import ASVEncoder
from aam.losses import PairwiseLoss
from aam.models.feedforward import FeedForward


@tf.keras.saving.register_keras_serializable(package="NucleotideEncoderV6")
class NucleotideEncoderV6(tf.keras.Model):
    def __init__(
        self,
        embedding_dim: int,
        max_bp: int,
        dropout_rate: float,
        intermediate_activation: str = "gelu",
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 256,
        **kwargs,
    ):
        super(NucleotideEncoderV6, self).__init__(**kwargs)
        print("Constructing V6 model")
        self.embedding_dim = embedding_dim
        self.max_bp = max_bp
        self.dropout_rate = dropout_rate
        self.intermediate_activation = intermediate_activation
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size

        self.loss_tracker = tf.keras.metrics.Mean()
        self.nuc_tracker = tf.keras.metrics.Mean()
        self.asv_loss = PairwiseLoss(use_mean_pairs=False)
        self.asv_tracker = tf.keras.metrics.Mean()

        self.asv_encoder = ASVEncoder(
            self.max_bp,
            self.attention_heads,
            self.attention_layers,
            self.dropout_rate,
            self.intermediate_size,
            intermediate_activation=self.intermediate_activation,
            embedding_dim=self.embedding_dim,
            name="asv_encoder",
        )
        self.asv_ff = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
                FeedForward(),
                tf.keras.layers.Dense(self.embedding_dim, dtype=tf.float32),
            ],
            name="asv_ff",
        )

    def build(self, input_shape):
        if self.built:
            return
        print("Building NucleotideEncoderV6...")
        self.asv_encoder.build(input_shape)
        input_shape = self.asv_encoder.compute_output_shape(input_shape)
        self.asv_ff.build(input_shape)
        super(NucleotideEncoderV6, self).build(input_shape)

    def build_graph(self, input_shape):
        """Builds graph

        Args:
            input_shape (tuple): A shape tuple (integers), not including the batch size.
        """
        super(NucleotideEncoderV6, self).build((None,) + input_shape)
        x = tf.keras.layers.Input(shape=(input_shape))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def predict_step(self, data):
        inputs, asv_ids = data
        return self(inputs, training=False), asv_ids

    def _compute_loss(self, y_true, embeddings):
        asv_loss = tf.reduce_mean(self.asv_loss(y_true, embeddings))
        return asv_loss

    def train_step(self, data):
        inputs, y_true = data
        with tf.GradientTape() as tape:
            embeddings = self(inputs, training=True)
            asv_loss = self._compute_loss(y_true, embeddings)
            nuc_loss = tf.reduce_sum(self.losses)
            unscaled_loss = asv_loss + nuc_loss
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(unscaled_loss)
            else:
                loss = unscaled_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(unscaled_loss)
        self.nuc_tracker.update_state(nuc_loss)
        self.asv_tracker.update_state(asv_loss)
        output_trackers = {
            "loss": self.loss_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "asv_loss": self.asv_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }
        return output_trackers

    def test_step(self, data):
        inputs, y_true = data

        embeddings = self(inputs, training=False)
        asv_loss = self._compute_loss(y_true, embeddings)
        nuc_loss = tf.reduce_sum(self.losses)
        loss = asv_loss + nuc_loss
        self.loss_tracker.update_state(loss)
        self.nuc_tracker.update_state(nuc_loss)
        self.asv_tracker.update_state(asv_loss)
        output_trackers = {
            "loss": self.loss_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "asv_loss": self.asv_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }
        return output_trackers

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable
        embeddings = self.asv_encoder(inputs, training=training)
        asv_embeddings = self.asv_ff(embeddings)
        return asv_embeddings

    def get_config(self):
        config = super(NucleotideEncoderV6, self).get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "max_bp": self.max_bp,
                "dropout_rate": self.dropout_rate,
                "intermediate_activation": self.intermediate_activation,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)

        if input_shape is not None:
            model.build([None, 150])
        return model
