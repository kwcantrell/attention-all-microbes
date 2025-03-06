from __future__ import annotations

from typing import Union
from aam.models.utils import sort_using_counts
import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="ConvolutionBlock")
class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, pool=False, **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool = pool

    def build(self, input_shape):
        self.conv_inner = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
        )
        if self.pool:
            self.conv_outer = tf.keras.layers.Conv1D(
                filters=self.filters * 2,
                kernel_size=self.kernel_size,
                strides=2,
                padding="same",
            )
            self.res_pool = tf.keras.layers.Conv1D(
                filters=self.filters * 2, kernel_size=1, strides=2, padding="same"
            )
        else:
            self.conv_outer = tf.keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=1,
                padding="same",
            )
        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )
        self.activation = tf.keras.layers.Activation("gelu")

    def call(self, inputs, training=False):
        output = self.conv_inner(inputs)
        output = self.activation(output)
        output = self.conv_outer(output)
        output = self.activation(output)
        if self.pool:
            inputs = self.res_pool(inputs)
        residual = inputs + self._rezero * output
        return residual

    def get_config(self):
        config = super(ConvolutionBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool": self.pool,
            }
        )
        return config


@tf.keras.saving.register_keras_serializable(package="AutoEncoder")
class AutoEncoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.built:
            print("AutoEncoder is already built")
            return

        asv_embeddings, taxonomy_counts = input_shape
        rarefy_depth = taxonomy_counts[1]
        emb_dim = asv_embeddings[-1]
        filters = 1
        kernel_size = 5
        taxonomy_conv_layers = [tf.keras.layers.Input([rarefy_depth, 1])]
        num_layers = 4
        for _ in range(num_layers):
            taxonomy_conv_layers += [
                ConvolutionBlock(filters, kernel_size, pool=False),
                ConvolutionBlock(filters, kernel_size, pool=False),
                ConvolutionBlock(filters, kernel_size, pool=True),
            ]
            filters *= 2
        self.taxonomy_encoder = tf.keras.Sequential(
            taxonomy_conv_layers + [tf.keras.layers.Dense(asv_embeddings[-1])]
        )
        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )
        filters = 1
        kernel_size = 5
        encoder_conv_layers = [tf.keras.layers.Input([emb_dim, 1])]
        for _ in range(5):
            encoder_conv_layers += [
                ConvolutionBlock(filters, kernel_size, pool=False),
                ConvolutionBlock(filters, kernel_size, pool=False),
                ConvolutionBlock(filters, kernel_size, pool=False),
                ConvolutionBlock(filters, kernel_size, pool=True),
            ]
            filters *= 2
        self.encoder = tf.keras.Sequential(encoder_conv_layers)
        super(AutoEncoder, self).build(input_shape)

    def call(
        self,
        inputs,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable
        asv_embeddings, taxonomy_counts = inputs
        taxonomy_embeddings = self.taxonomy_encoder(taxonomy_counts, training=training)
        taxonomy_embeddings = tf.reduce_mean(taxonomy_embeddings, axis=1)
        encoder_input = asv_embeddings + self._rezero * taxonomy_embeddings
        intermediate_embeddings = self.encoder(encoder_input, training=training)

        print("Triplet AutoEncoder exit...")
        return intermediate_embeddings

    def get_config(self):
        config = super(AutoEncoder, self).get_config()
        return config


@tf.keras.saving.register_keras_serializable(package="SequenceRegressorV2")
class SequenceRegressorV2(tf.keras.Model):
    def __init__(self, rarefy_depth, **kwargs):
        super(SequenceRegressorV2, self).__init__(**kwargs)
        self.rarefy_depth = rarefy_depth

    def build(self, input_shape):
        if self.built:
            return
        embeddings, batch_indices, asv_indices, counts = input_shape
        self.encoder = AutoEncoder(name="auto_encoder")
        self.encoder.build([[None, embeddings[-1]], [None, self.rarefy_depth]])

        emb_dim = embeddings[-1]
        filters = 32
        kernel_size = 3
        enc = emb_dim // (2**5)
        classifier_layers = [tf.keras.layers.Input([enc, 32])]
        for _ in range(12):
            classifier_layers += [ConvolutionBlock(filters, kernel_size, pool=False)]
        self.classifier = tf.keras.Sequential(
            classifier_layers
            + [
                tf.keras.layers.Dense(1, use_bias=True),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, use_bias=True),
            ]
        )
        super(SequenceRegressorV2, self).build(input_shape)

    def batch_embeddings(
        self, asv_embeddings, batch_indicies, counts, asv_indices=None
    ):
        emb_dim = tf.shape(asv_embeddings)[-1]
        batch_indicies = tf.cast(batch_indicies, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        if asv_indices is not None:
            asv_embeddings = tf.gather(asv_embeddings, asv_indices)
        batch_shape = tf.reduce_max(batch_indicies[:, 0]) + 1
        max_unique = tf.reduce_max(batch_indicies[:, 1]) + 1
        batch_embeddings = tf.scatter_nd(
            batch_indicies, asv_embeddings, shape=[batch_shape, max_unique, emb_dim]
        )
        counts = tf.scatter_nd(
            batch_indicies, counts, shape=[batch_shape, max_unique, 1]
        )
        batch_embeddings, counts = sort_using_counts(batch_embeddings, counts)
        return batch_embeddings, counts

    def call(
        self, inputs, training: bool = False
    ) -> Union[
        tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        training = training and self.trainable
        if len(inputs) == 4:
            tokens, batch_indices, asv_indices, counts = inputs
            asv_embeddings, counts = self.batch_embeddings(
                tokens, batch_indices, counts, asv_indices
            )
            mask = tf.cast(counts > 0, dtype=self.compute_dtype)
            asv_embeddings = tf.cast(asv_embeddings, dtype=self.compute_dtype) * mask
        else:
            asv_embeddings, counts = inputs
            mask = tf.cast(counts > 0, dtype=tf.float32)

        seq_dim = tf.shape(asv_embeddings)[1]
        asv_embeddings = tf.reduce_sum(asv_embeddings, axis=1) / tf.reduce_sum(
            mask, axis=1
        )
        counts = tf.ensure_shape(counts, [None, None, 1])
        counts = tf.cast(counts, dtype=tf.float32) / tf.cast(
            self.rarefy_depth, dtype=tf.float32
        )
        counts = tf.pad(
            counts, paddings=[[0, 0], [0, self.rarefy_depth - seq_dim], [0, 0]]
        )
        encoder_output = self.encoder([asv_embeddings, counts])
        output = self.classifier(encoder_output)
        return output, output

    def get_config(self):
        config = super(SequenceRegressorV2, self).get_config()
        config.update(
            {
                "build_input_shape": self.get_build_config(),
                "rarefy_depth": self.rarefy_depth,
            }
        )

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)

        if input_shape is not None:
            model.build(input_shape)

        return model
