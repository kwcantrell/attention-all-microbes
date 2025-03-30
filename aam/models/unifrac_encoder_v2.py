from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.losses import PairwiseLoss
from aam.models.feedforward import FeedForward

# from aam.models.conv_feedforward_v2 import ConvFeedForwardV2
from aam.models.transformers import TransformerEncoder

# @tf.keras.saving.register_keras_serializable(package="UnifracEncoderV2")
# class UnifracEncoderV2(tf.keras.Model):
#     def __init__(
#         self,
#         num_encoder_layers=12,
#         num_filters=32,
#         kernel_size=3,
#         conv_blocks_per_layer=8,
#         **kwargs,
#     ):
#         super(UnifracEncoderV2, self).__init__(**kwargs)
#         self.unifrac_loss = PairwiseLoss(use_mean_pairs=False)
#         self.loss_tracker = tf.keras.metrics.Mean(name="loss")
#         print("Num layers:", num_encoder_layers)
#         print("blocks per layers:", conv_blocks_per_layer)
#         self.num_encoder_layers = num_encoder_layers
#         self.num_filters = num_filters
#         self.kernel_size = kernel_size
#         self.conv_blocks_per_layer = conv_blocks_per_layer

#     def build(self, input_shape):
#         if self.built:
#             print("UnifracEncoderV2 is already built")
#             return
#         sparse_indices, embeddings = input_shape
#         self.encoder = TransformerEncoder(
#             num_layers=8,
#             num_attention_heads=4,
#             intermediate_size=1024,
#             use_linear_bias=True,
#         )
#         self.ff = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Dense(embeddings[-1], activation="gelu"),
#             ]
#         )
#         self._rezero = self.add_weight(
#             name="rezero",
#             dtype=tf.float32,
#             initializer=tf.keras.initializers.Zeros(),
#             trainable=True,
#         )
#         self.output_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
#         super(UnifracEncoderV2, self).build(input_shape)

#     def predict_step(
#         self,
#         data: Union[
#             tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
#             tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
#         ],
#     ):
#         inputs, y = data
#         return self(inputs, training=False), y

#     def _compute_loss(self, y, output_embeddings):
#         loss = self.unifrac_loss(y, output_embeddings)
#         return tf.reduce_mean(loss)

#     def train_step(
#         self,
#         data: Union[
#             tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
#             tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
#         ],
#     ):
#         inputs, y = data

#         with tf.GradientTape() as tape:
#             output_embeddings = self(inputs, training=True)
#             unscaled_loss = self._compute_loss(y, output_embeddings)
#             if self.compute_dtype == "float16":
#                 loss = self.optimizer.get_scaled_loss(unscaled_loss)
#             else:
#                 loss = unscaled_loss
#         gradients = tape.gradient(loss, self.trainable_variables)
#         if self.compute_dtype == "float16":
#             gradients = self.optimizer.get_unscaled_gradients(gradients)
#         self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

#         self.loss_tracker.update_state(unscaled_loss)
#         return {"loss": self.loss_tracker.result()}

#     def test_step(
#         self,
#         data: Union[
#             tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
#             tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
#         ],
#     ):
#         inputs, y = data
#         output_embeddings = self(inputs, training=False)
#         loss = self._compute_loss(y, output_embeddings)

#         self.loss_tracker.update_state(loss)
#         return {"loss": self.loss_tracker.result()}

#     def _get_dense_embeddings(self, inputs):
#         sparse_indices, embeddings = inputs
#         sparse_indices = tf.cast(sparse_indices, dtype=tf.int32)
#         batch_dim = tf.reduce_max(sparse_indices[:, 0]) + 1
#         seq_dim = tf.reduce_max(sparse_indices[:, 1]) + 1
#         dense_embeddings = tf.scatter_nd(
#             sparse_indices,
#             embeddings,
#             [batch_dim, seq_dim, tf.shape(embeddings)[-1]],
#         )
#         mask = tf.scatter_nd(
#             sparse_indices,
#             tf.ones([tf.shape(sparse_indices)[0], 1], dtype=self.compute_dtype),
#             [batch_dim, seq_dim, 1],
#         )
#         return dense_embeddings, mask

#     def call(
#         self, inputs, training: bool = False
#     ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
#         dense_embeddings, mask = self._get_dense_embeddings(inputs)
#         dense_embeddings = tf.cast(dense_embeddings, dtype=self.compute_dtype)

#         encoder_output = self.encoder(dense_embeddings, mask=mask, training=training)
#         ff_input = tf.reduce_sum(encoder_output, axis=1) / tf.reduce_sum(mask, axis=1)

#         output = ff_input + tf.cast(self._rezero, dtype=self.compute_dtype) * self.ff(
#             ff_input
#         )

#         print("UnifracEncoderV2 exit...")
#         return self.output_activation(output)

#     def get_config(self):
#         config = super(UnifracEncoderV2, self).get_config()
#         config.update(
#             {
#                 "num_encoder_layers": self.num_encoder_layers,
#                 "num_filters": self.num_filters,
#                 "kernel_size": self.kernel_size,
#                 "conv_blocks_per_layer": self.conv_blocks_per_layer,
#                 "build_input_shape": self.get_build_config(),
#             }
#         )
#         return config

#     @classmethod
#     def from_config(cls, config):
#         print("Constructing UnifracEncoderV2 from config")
#         input_shape = None
#         if "build_input_shape" in config:
#             build_input_shape = config.pop("build_input_shape")
#             input_shape = build_input_shape["input_shape"]

#         model = cls(**config)
#         model.build(input_shape)
#         return model


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
        print("Num layers:", num_encoder_layers)
        print("blocks per layers:", conv_blocks_per_layer)
        self.num_encoder_layers = num_encoder_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_blocks_per_layer = conv_blocks_per_layer

    def build(self, input_shape):
        if self.built:
            print("UnifracEncoderV2 is already built")
            return
        sparse_indices, embeddings = input_shape
        self.encoder = TransformerEncoder(
            num_layers=8,
            num_attention_heads=4,
            intermediate_size=1024,
            use_linear_bias=True,
        )
        self.ff = tf.keras.Sequential(
            [FeedForward(), tf.keras.layers.Dense(embeddings[-1])]
        )
        self.output_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
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
            unscaled_loss = self._compute_loss(y, output_embeddings)
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(unscaled_loss)
            else:
                loss = unscaled_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(unscaled_loss)
        return {"loss": self.loss_tracker.result()}

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
        return {"loss": self.loss_tracker.result()}

    def _get_dense_embeddings(self, inputs):
        sparse_indices, embeddings = inputs
        sparse_indices = tf.cast(sparse_indices, dtype=tf.int32)
        batch_dim = tf.reduce_max(sparse_indices[:, 0]) + 1
        seq_dim = tf.reduce_max(sparse_indices[:, 1]) + 1
        dense_embeddings = tf.scatter_nd(
            sparse_indices,
            embeddings,
            [batch_dim, seq_dim, tf.shape(embeddings)[-1]],
        )
        mask = tf.scatter_nd(
            sparse_indices,
            tf.ones([tf.shape(sparse_indices)[0], 1], dtype=self.compute_dtype),
            [batch_dim, seq_dim, 1],
        )
        return dense_embeddings, mask

    def call(
        self, inputs, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        dense_embeddings, mask = self._get_dense_embeddings(inputs)
        dense_embeddings = tf.cast(dense_embeddings, dtype=self.compute_dtype)

        encoder_output = self.encoder(dense_embeddings, mask=mask, training=training)
        ff_input = tf.reduce_sum(encoder_output, axis=1) / tf.reduce_sum(mask, axis=1)

        output = self.ff(ff_input)

        print("UnifracEncoderV2 exit...")
        return self.output_activation(output)

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
