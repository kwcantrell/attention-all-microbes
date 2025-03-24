from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.models.regressor_v2 import DenseBlock


@tf.keras.saving.register_keras_serializable(package="RegressorV3")
class RegressorV3(tf.keras.Model):
    def __init__(self, base_model, shift, scale, num_encoder_layers=6, **kwargs):
        super(RegressorV3, self).__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_tracker = tf.keras.metrics.Mean(name="mae")

        self.num_encoder_layers = num_encoder_layers
        self.shift = shift
        self.scale = scale
        self.base_model = base_model
        self.base_model.trainable = False

    def build(self, input_shape):
        if self.built:
            print("RegressorV3 is already built")
            return
        self.base_norm = tf.keras.layers.BatchNormalization()
        layers = []
        for _ in range(3):
            layers.append(DenseBlock(pool=True))
        self.regressor = tf.keras.Sequential(
            layers + [tf.keras.layers.Dense(1)], name="regressor"
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
        base_embeddings = self.base_norm(base_embeddings, training=training)

        output = self.regressor(base_embeddings)
        print("RegressorV3 exit...")
        return output

    def get_config(self):
        config = super(RegressorV3, self).get_config()
        config.update(
            {
                "shift": self.shift,
                "scale": self.scale,
                "num_encoder_layers": self.num_encoder_layers,
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

        model = cls(**config)
        model.build(input_shape)
        return model
