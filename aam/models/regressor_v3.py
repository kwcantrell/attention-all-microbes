from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.models.conv_feedforward import ConvFeedForward
from aam.models.convolution_block import ConvolutionBlock
from aam.models.feedforward import FeedForward


@tf.keras.saving.register_keras_serializable(package="RegressorV3")
class RegressorV3(tf.keras.Model):
    def __init__(
        self,
        base_model,
        shift,
        scale,
        num_filters=32,
        kernel_size=3,
        num_layers=6,
        conv_blocks_per_layer=8,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(RegressorV3, self).__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_tracker = tf.keras.metrics.Mean(name="mae")

        self.num_layers = num_layers
        self.shift = shift
        self.scale = scale
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.base_model = base_model
        self.base_model.trainable = False
        self.conv_blocks_per_layer = conv_blocks_per_layer
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        if self.built:
            print("RegressorV3 is already built")
            return
        layers = []
        for _ in range(self.num_layers):
            layers += [
                ConvFeedForward(
                    self.num_filters,
                    self.kernel_size,
                    num_layers=self.conv_blocks_per_layer,
                )
            ]
        self.regressor = tf.keras.Sequential(
            layers + [FeedForward(outdim=1)],
            name="regressor",
        )
        super(RegressorV3, self).build(input_shape)

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

    def call(self, inputs, training=False):
        base_embeddings = self.base_model(inputs, training=False)

        output = self.regressor(base_embeddings)
        print("RegressorV3 exit...")
        return output

    def get_config(self):
        config = super(RegressorV3, self).get_config()
        config.update(
            {
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "shift": self.shift,
                "scale": self.scale,
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "conv_blocks_per_layer": self.conv_blocks_per_layer,
                "num_layers": self.num_layers,
                "dropout_rate": self.dropout_rate,
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        print("Constructing RegressorV3 from config")
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]
        config["base_model"] = tf.keras.saving.deserialize_keras_object(
            config["base_model"]
        )
        model = cls(**config)
        model.build(input_shape)
        return model
