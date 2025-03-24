from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.models.asv_dense_count_encoder import ASVDenseCountEncoder
from aam.models.utils import sample_embeddings


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, pool=False, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.pool = pool

    def build(self, input_shape):
        units = input_shape[-1]
        if self.pool:
            self.dense_block = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(units, activation="gelu"),
                    tf.keras.layers.Dense(units // 2),
                ]
            )
        else:
            self.dense_block = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(units, activation="gelu"),
                    tf.keras.layers.Dense(units),
                ]
            )

        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, inputs, training=False):
        output = self.dense_block(inputs)

        # residual step
        output = inputs + self._rezero * output
        return output


@tf.keras.saving.register_keras_serializable(package="RegressorV2")
class RegressorV2(tf.keras.Model):
    def __init__(self, shift, scale, num_encoder_layers=1, **kwargs):
        super(RegressorV2, self).__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_tracker = tf.keras.metrics.Mean(name="mae")

        self.num_encoder_layers = num_encoder_layers
        self.shift = shift
        self.scale = scale

    def build(self, input_shape):
        if self.built:
            print("RegressorV2 is already built")
            return

        self.asv_dense_encoder = ASVDenseCountEncoder()
        self.regressor = tf.keras.Sequential(
            [
                DenseBlock(pool=True),
                DenseBlock(pool=True),
                DenseBlock(pool=True),
                tf.keras.layers.Dense(1, name="regressor"),
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

    def call(self, inputs, training=False):
        asv_embeddings, batch_indices, asv_indices, asv_counts, dense_counts = inputs

        batch_indices = tf.cast(batch_indices, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)

        asv_embeddings = sample_embeddings(
            asv_embeddings, batch_indices, asv_counts, asv_indices
        )
        asv_dense_embeddings = self.asv_dense_encoder(
            [asv_embeddings, dense_counts], training=training
        )

        output = self.regressor(asv_dense_embeddings)
        print("RegressorV2 exit...")
        return output

    def get_config(self):
        config = super(RegressorV2, self).get_config()
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
        print("Constructing RegressorV2 from config")
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)
        model.build(input_shape)
        return model
