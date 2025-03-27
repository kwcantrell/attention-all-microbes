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
        conv_blocks_per_layers: int = 1,
        num_nuc_layers: int = 3,
        num_asv_layers: int = 3,
        **kwargs,
    ):
        super(ASVEncoderV3, self).__init__(**kwargs)
        print("Constructing ASVEncoderV3")
        self.filters = filters

        self.kernel_size = kernel_size
        self.conv_blocks_per_layers = conv_blocks_per_layers
        self.num_nuc_layers = num_nuc_layers
        self.num_asv_layers = num_asv_layers
        self.base_tokens = 5

        self.loss_tracker = tf.keras.metrics.Mean()
        self.asv_loss = PairwiseLoss(use_mean_pairs=False)

    def build(self, input_shape):
        if self.built:
            return

        self.emb_layer = tf.keras.layers.Embedding(
            self.base_tokens,
            self.filters,
            input_length=input_shape[-1],
            embeddings_initializer="glorot_uniform",
        )

        nuc_block = []
        for _ in range(self.num_nuc_layers):
            nuc_block += [
                ConvolutionBlock(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    num_blocks=self.conv_blocks_per_layers,
                )
            ]
        self.nuc_block = tf.keras.Sequential(nuc_block, name="nuc_block")

        asv_layers = []
        for _ in range(self.num_asv_layers - 1):
            asv_layers += [
                ConvFeedForwardV2(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    conv_blocks=self.conv_blocks_per_layers,
                )
            ]
        self.asv_encoder = tf.keras.Sequential(
            asv_layers
            + [
                ConvFeedForwardV2(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    conv_blocks=self.conv_blocks_per_layers,
                    outdim=self.filters,
                )
            ]
        )
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
        inputs = self.emb_layer(inputs)

        nuc_output = self.nuc_block(inputs)
        asv_input = tf.reduce_mean(nuc_output, axis=-1)

        return self.asv_encoder(asv_input)

    def get_config(self):
        config = super(ASVEncoderV3, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "conv_blocks_per_layers": self.conv_blocks_per_layers,
                "num_asv_layers": self.num_asv_layers,
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
