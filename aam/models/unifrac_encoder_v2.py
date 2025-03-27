from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.losses import PairwiseLoss
from aam.models.conv_feedforward_v2 import ConvFeedForwardV2
from aam.models.convolution_block import ConvolutionBlock
from aam.models.utils import batch_embeddings


@tf.keras.saving.register_keras_serializable(package="UnifracEncoderV2")
class UnifracEncoderV2(tf.keras.Model):
    def __init__(
        self,
        num_encoder_layers=12,
        num_filters=32,
        kernel_size=3,
        conv_blocks_per_layer=8,
        **kwargs,
    ):
        super(UnifracEncoderV2, self).__init__(**kwargs)
        self.unifrac_loss = PairwiseLoss(use_mean_pairs=False)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.num_encoder_layers = num_encoder_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_blocks_per_layer = conv_blocks_per_layer

    def build(self, input_shape):
        if self.built:
            print("UnifracEncoderV2 is already built")
            return
        asv_embeddings, batch_indices, asv_indices, asv_counts, dense_counts = (
            input_shape
        )
        asv_layers = []
        for _ in range(self.num_encoder_layers):
            asv_layers += [
                ConvolutionBlock(
                    self.num_filters,
                    self.kernel_size,
                    strides=self.kernel_size,
                    num_blocks=1,
                )
            ]
        self.asv_encoder = tf.keras.Sequential(
            asv_layers + [tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))],
            name="asv_encoder",
        )
        self.asv_encoder.build([None, dense_counts[-1], asv_embeddings[-1]])

        encoder_layers = []
        for _ in range(self.num_encoder_layers):
            encoder_layers += [
                ConvFeedForwardV2(
                    self.num_filters,
                    self.kernel_size,
                    conv_blocks=self.conv_blocks_per_layer,
                )
            ]
        self.encoder = tf.keras.Sequential(encoder_layers, name="encoder")
        self.encoder.build([None, asv_embeddings[-1]])
        super(UnifracEncoderV2, self).build(input_shape)

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        return self(inputs, training=False), y

    def _compute_loss(self, y, output_embeddings):
        loss = self.unifrac_loss(y, output_embeddings)
        return tf.reduce_mean(loss)

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data

        with tf.GradientTape() as tape:
            output_embeddings = self(inputs, training=True)
            loss = self._compute_loss(y, output_embeddings)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def test_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        output_embeddings = self(inputs, training=False)
        loss = self._compute_loss(y, output_embeddings)

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def call(
        self, inputs, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        asv_embeddings, batch_indices, asv_indices, asv_counts, dense_counts = inputs

        batch_indices = tf.cast(batch_indices, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)
        asv_embeddings, _ = batch_embeddings(
            asv_embeddings, batch_indices, asv_counts, asv_indices
        )
        encoder_input = self.asv_encoder(asv_embeddings)
        output_embeddings = self.encoder(encoder_input)
        print("UnifracEncoderV2 exit...")
        return output_embeddings

    def get_config(self):
        config = super(UnifracEncoderV2, self).get_config()
        config.update(
            {
                "num_encoder_layers": self.num_encoder_layers,
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "conv_blocks_per_layer": self.conv_blocks_per_layer,
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        print("Constructing UnifracEncoderV2 from config")
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)
        model.build(input_shape)
        return model
