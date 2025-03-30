from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.models.convolution_pooler_v2 import NonPoolingBlock, PoolingBlock
from aam.models.feedforward import FeedForward
from aam.models.transformers import TransformerEncoder


@tf.keras.saving.register_keras_serializable(package="RegressorV2")
class RegressorV2(tf.keras.Model):
    def __init__(
        self,
        shift,
        scale,
        base_mode,
        num_filters=256,
        kernel_size=3,
        pooling_size=1024,
        **kwargs,
    ):
        super(RegressorV2, self).__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_tracker = tf.keras.metrics.Mean(name="mae")

        self.shift = shift
        self.scale = scale
        base_mode.trainable = False
        self.base_model = base_mode
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size

    def build(self, input_shape):
        if self.built:
            print("RegressorV2 is already built")
            return
        self.count_extractor = tf.keras.Sequential(
            [
                tf.keras.layers.Reshape([-1, 1]),
                PoolingBlock(self.num_filters, self.pooling_size),
            ],
            name="count_extractor",
        )
        self.encoder = TransformerEncoder(intermediate_size=512, name="encoder")
        self._rezero = self.add_weight(
            name="rezero",
            dtype=tf.float32,
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.regressor = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
                FeedForward(),
                tf.keras.layers.Dense(units=1),
            ],
            name="regressor",
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
        sparse_indices, sample_embeddings, dense_counts = inputs
        sample_embeddings = self.base_model(
            [sparse_indices, sample_embeddings], training=training
        )

        dense_counts = tf.math.log1p(
            dense_counts / tf.reduce_sum(dense_counts, axis=-1, keepdims=True)
        )
        extractor_output = self.count_extractor(dense_counts, training=training)
        sample_embeddings /= tf.cast(tf.shape(extractor_output)[1], dtype=tf.float32)
        sample_embeddings = tf.expand_dims(sample_embeddings, axis=1)
        encoder_input = sample_embeddings + self._rezero * extractor_output

        encoder_output = self.encoder(encoder_input)
        output_embeddings = self.regressor(encoder_output)
        print("RegressorV2 exit...")
        return output_embeddings

    def get_config(self):
        config = super(RegressorV2, self).get_config()
        config.update(
            {
                "shift": self.shift,
                "scale": self.scale,
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        print("Constructing RegressorV2 from config")
        input_shape = None
        base_model = tf.keras.saving.deserialize_keras_object(config["base_model"])
        base_model.trainable = False
        config["base_model"] = base_model
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)
        model.build(input_shape)
        return model
