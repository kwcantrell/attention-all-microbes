from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.losses import PairwiseLoss
from aam.models.asv_dense_count_encoder_v3 import ASVDenseCountEncoderV3
from aam.models.conv_feedforward import ConvFeedForward
from aam.models.utils import sample_embeddings


@tf.keras.saving.register_keras_serializable(package="RegressorV2")
class RegressorV2(tf.keras.Model):
    def __init__(
        self,
        shift,
        scale,
        num_encoder_layers=8,
        non_pool_blocks_per_layer=2,
        num_filters=32,
        kernel_size=3,
        conv_blocks_per_layer=2,
        **kwargs,
    ):
        super(RegressorV2, self).__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_tracker = tf.keras.metrics.Mean(name="mae")

        self.shift = shift
        self.scale = scale
        self.num_encoder_layers = num_encoder_layers
        self.non_pool_blocks_per_layer = non_pool_blocks_per_layer
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_blocks_per_layer = conv_blocks_per_layer

    def build(self, input_shape):
        if self.built:
            print("RegressorV2 is already built")
            return

        self.asv_encoder = ASVDenseCountEncoderV2(name="asv_encoder")

        encoder_layers = []
        for _ in range(self.num_encoder_layers):
            encoder_layers += [
                ConvFeedForward(
                    self.num_filters,
                    self.kernel_size,
                    conv_blocks=self.conv_blocks_per_layer,
                )
            ]
        self.encoder = tf.keras.Sequential(
            encoder_layers
            + [
                ConvFeedForward(
                    self.num_filters,
                    self.kernel_size,
                    conv_blocks=self.conv_blocks_per_layer,
                    outdim=1,
                )
            ]
        )
        super(RegressorV2, self).build(input_shape)

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        output = self(inputs, training=False)
        y = y * self.scale + self.shift
        output = output * self.scale + self.shift
        return output, y

    def _compute_loss(self, y, output):
        loss = tf.reduce_mean(tf.square(y - output))
        return loss

    def _compute_metric(self, y, output):
        y = y * self.scale + self.shift
        output = output * self.scale + self.shift
        mae = tf.reduce_mean(tf.abs(y - output))
        return mae

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data

        with tf.GradientTape() as tape:
            output = self(inputs, training=True)
            loss = self._compute_loss(y, output)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        mae = self._compute_metric(y, output)
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(mae)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result(),
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
        output = self(inputs, training=False)
        loss = self._compute_loss(y, output)
        mae = self._compute_metric(y, output)
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(mae)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def call(
        self, inputs, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        encoder_input = self.asv_encoder(inputs)
        output_embeddings = self.encoder(encoder_input)
        print("RegressorV2 exit...")
        return output_embeddings

    def get_config(self):
        config = super(RegressorV2, self).get_config()
        config.update(
            {
                "shift": self.shift,
                "scale": self.scale,
                "num_encoder_layers": self.num_encoder_layers,
                "non_pool_blocks_per_layer": self.non_pool_blocks_per_layer,
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "conv_blocks_per_layer": self.conv_blocks_per_layer,
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        print("Constructing RegressorV2 from config")
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)
        model.build(input_shape)
        return model
