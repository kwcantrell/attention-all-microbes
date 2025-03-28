from __future__ import annotations

import tensorflow as tf

from aam.losses import PairwiseLoss
from aam.models.conv_feedforward_v2 import ConvFeedForwardV2
from aam.models.convolution_block import ConvolutionBlock


@tf.keras.saving.register_keras_serializable(package="ASVEncoderV3")
class ASVEncoderV3(tf.keras.Model):
    def __init__(
        self,
        filters: int = 32,
        kernel_size: int = 3,
        conv_blocks_per_layers: int = 8,
        num_layers: int = 6,
        **kwargs,
    ):
        super(ASVEncoderV3, self).__init__(**kwargs)
        print("Constructing ASVEncoderV3")
        self.filters = filters

        self.kernel_size = kernel_size
        self.conv_blocks_per_layers = conv_blocks_per_layers
        self.num_layers = num_layers
        self.base_tokens = 5

        self.loss_tracker = tf.keras.metrics.Mean()
        self.asv_loss = PairwiseLoss(use_mean_pairs=False)

    def build(self, input_shape):
        if self.built:
            return

        seq_size = input_shape[-1]
        num_tokens = self.base_tokens * seq_size
        self.nucleotide_position = tf.reshape(
            tf.range(0, self.base_tokens * seq_size, self.base_tokens, dtype=tf.int32),
            shape=[1, -1],
        )

        self.emb_layer = tf.keras.layers.Embedding(
            num_tokens, self.filters, input_length=input_shape[-1]
        )

        layers = []
        for i in range(self.num_layers):
            layers += [
                ConvFeedForwardV2(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    conv_blocks=self.conv_blocks_per_layers,
                )
            ]
        layers.append(tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)))
        self.encoder = tf.keras.Sequential(layers, name="encoder")

        super(ASVEncoderV3, self).build(input_shape)

    def predict_step(self, data):
        inputs, asv_ids = data
        return self(inputs, training=False), asv_ids

    def _compute_loss(self, y_true, embeddings):
        loss = tf.reduce_mean(self.asv_loss(y_true, embeddings))
        return loss

    def train_step(self, data):
        inputs, y_true = data
        with tf.GradientTape() as tape:
            embeddings = self(inputs, training=True)
            loss = self._compute_loss(y_true, embeddings)

            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        output_trackers = {
            "loss": self.loss_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }
        return output_trackers

    def test_step(self, data):
        inputs, y_true = data

        embeddings = self(inputs, training=False)
        loss = self._compute_loss(y_true, embeddings)

        self.loss_tracker.update_state(loss)
        output_trackers = {
            "loss": self.loss_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }
        return output_trackers

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, dtype=tf.int32)
        inputs += self.nucleotide_position
        nuc_input = self.emb_layer(inputs)
        return self.encoder(nuc_input)

    def get_config(self):
        config = super(ASVEncoderV3, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "conv_blocks_per_layers": self.conv_blocks_per_layers,
                "num_layers": self.num_layers,
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

        model.build(input_shape)
        return model
