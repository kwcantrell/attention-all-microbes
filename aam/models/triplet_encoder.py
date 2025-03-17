from __future__ import annotations

import math
from typing import Union

import tensorflow as tf
import tensorflow_addons as tfa

from aam.callbacks import LAMBLRScheduler

# from aam.data_handlers.generator_dataset import batch_embeddings
from aam.losses import _pairwise_distances, global_orthogonal_regulization
from aam.models.unifrac_encoder import UnifracEncoder
from aam.models.utils import cos_decay_with_warmup, sort_using_counts
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler


def _proj(u, v):
    """project u onto v"""
    u = tf.cast(u, dtype=tf.float64)
    v = tf.cast(v, dtype=tf.float64)
    v_norm = tf.sqrt(tf.reduce_sum(v * v, axis=-1, keepdims=True))
    unit_v = v / v_norm
    proj = tf.reduce_sum(u * unit_v, axis=-1, keepdims=True) * unit_v
    proj = tf.cast(proj, dtype=tf.float32)
    return proj


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, pool=False, **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool = pool

        self.conv_inner = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
        )

        if self.pool:
            self.conv_outer = tf.keras.layers.Conv1D(
                filters=self.filters * 2,
                kernel_size=2,
                strides=2,
                padding="same",
            )
            self.res_pool = tf.keras.layers.Conv1D(
                filters=self.filters * 2,
                kernel_size=2,
                strides=2,
                padding="same",
            )
        else:
            self.conv_outer = tf.keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=1,
                padding="same",
            )

        self.activation = tf.keras.layers.Activation("gelu")
        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, inputs, training=False):
        # first block
        output = self.conv_inner(inputs)
        output = self.activation(output)

        # second block
        output = self.conv_outer(output)
        output = self.activation(output)
        if self.pool:
            inputs = self.res_pool(inputs)

        # residual connection
        return inputs + self._rezero * output

    def get_config(self):
        config = super(ConvolutionBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool": self.pool,
            }
        )


class UpscaleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(UpscaleBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

        self.conv_inner = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
        )

        self.conv_outer = tf.keras.layers.Conv1D(
            filters=self.filters // 2,
            kernel_size=1,
            strides=1,
            padding="same",
        )

        self.scaler = tf.keras.layers.UpSampling1D(size=2)
        self.res_scale = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=self.filters // 2,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                ),
                tf.keras.layers.UpSampling1D(size=2),
            ]
        )
        self.activation = tf.keras.layers.Activation("gelu")
        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, inputs, training=False):
        # first block
        output = self.conv_inner(inputs)
        output = self.activation(output)

        # second block
        output = self.conv_outer(output)
        output = self.activation(output)
        output = self.scaler(output)

        # residual connection
        inputs = self.res_scale(inputs)
        return inputs + self._rezero * output

    def get_config(self):
        config = super(UpscaleBlock, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})


@tf.keras.saving.register_keras_serializable(package="AutoEncoder")
class AutoEncoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.built:
            print("AutoEncoder is already built")
            return

        asv_embeddings, taxonomy_counts = input_shape
        emb_dim = asv_embeddings[-1]
        filters = 1
        kernel_size = 7
        taxonomy_conv_layers = [tf.keras.layers.Input([taxonomy_counts[-1], 1])]
        num_layers = int(math.log(512) / math.log(2))
        for _ in range(num_layers):
            taxonomy_conv_layers += [
                ConvolutionBlock(filters, kernel_size, pool=False),
                ConvolutionBlock(filters, kernel_size, pool=False),
                ConvolutionBlock(filters, kernel_size, pool=False),
                ConvolutionBlock(filters, kernel_size, pool=True),
            ]
            filters *= 2
        self.taxonomy_encoder = tf.keras.Sequential(taxonomy_conv_layers)
        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

        filters = 1
        kernel_size = 5
        encoder_conv_layers = [tf.keras.layers.Input([emb_dim, 1])]
        for _ in range(6):
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
        return intermediate_embeddings, encoder_input


@tf.keras.saving.register_keras_serializable(package="TripletEncoder")
class TripletEncoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(TripletEncoder, self).__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.asv_loss = _pairwise_distances
        self.asv_rec_tracker = tf.keras.metrics.Mean(name="asv_rec_loss")
        self.batch_noise_Tracker = tf.keras.metrics.Mean(name="batch_noise_loss")

        self.triplet_loss = global_orthogonal_regulization
        self.res_class_tracker = tf.keras.metrics.Mean(name="ortho_loss")
        self.discriminator_loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1
        )
        self.batch_mag_tracker = tf.keras.metrics.Mean(name="discriminator_mag")
        self.batch_class_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        self.age_tracker = tf.keras.metrics.Mean(name="age_loss")
        self.num_groups = 64

    def build(self, input_shape):
        if self.built:
            print("TripletEncoder is already built")
            return

        asv_embeddings, batch_indices, asv_indices, asv_counts, taxonomy_counts = (
            input_shape
        )
        self.encoder = AutoEncoder(name="auto_encoder")
        self.encoder.build([asv_embeddings, taxonomy_counts])

        filters = 64
        kernel_size = 3
        discriminator_in_layers = [tf.keras.layers.Input([8, 64])]
        for _ in range(12):
            discriminator_in_layers += [
                ConvolutionBlock(filters, kernel_size, pool=False)
            ]
        self.discriminator_in = tf.keras.Sequential(discriminator_in_layers)

        self.batch_classifier = tf.keras.layers.Dense(
            self.num_groups, use_bias=True, activation="softmax"
        )
        self.regressor = tf.keras.layers.Dense(1)

        # def _ff_block(output_dim, use_bias=True):
        #     block = [
        #         tf.keras.layers.Dense(
        #             output_dim,
        #             use_bias=use_bias,
        #             kernel_initializer=tf.keras.initializers.HeUniform(),
        #         ),
        #         tf.keras.layers.BatchNormalization(dtype=tf.float32),
        #         tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x)),
        #     ]
        #     return block

        # decoder_layers = []
        # out_dim = 64
        # while out_dim < 512:
        #     decoder_layers += _ff_block(output_dim=out_dim, use_bias=True)
        #     decoder_layers += _ff_block(output_dim=out_dim, use_bias=True)
        #     out_dim *= 2
        # self.decoder = tf.keras.Sequential(
        #     decoder_layers + [tf.keras.layers.Dense(512)]
        # )

        # filters = 512
        # kernel_size = 1
        # discriminator_out_layers = [tf.keras.layers.Input([1, 512])]
        # for _ in range(3):
        #     discriminator_out_layers += [
        #         ConvolutionBlock(filters, kernel_size, pool=False),
        #         ConvolutionBlock(filters, kernel_size, pool=False),
        #         ConvolutionBlock(filters, kernel_size, pool=False),
        #         UpscaleBlock(filters, kernel_size),
        #     ]
        #     filters /= 2
        #     if kernel_size == 1:
        #         kernel_size += 1
        # self.discriminator_out = tf.keras.Sequential(discriminator_out_layers)

        filters = 64
        kernel_size = 5
        decoder_conv_layers = [
            tf.keras.layers.Input([8, 64]),
        ]
        for _ in range(6):
            decoder_conv_layers += [
                ConvolutionBlock(filters, kernel_size, pool=False),
                ConvolutionBlock(filters, kernel_size, pool=False),
                ConvolutionBlock(filters, kernel_size, pool=False),
                UpscaleBlock(filters, kernel_size),
            ]
            filters /= 2
        self.decoder = tf.keras.Sequential(
            decoder_conv_layers
            + [tf.keras.layers.Flatten(), tf.keras.layers.Dense(512)]
        )

        super(TripletEncoder, self).build(input_shape)

    def _reconstruction_loss(self, encoder_input, decodeer_output):
        square_difference = tf.reduce_sum(
            tf.square(encoder_input - decodeer_output), axis=-1
        )
        return tf.reduce_mean(square_difference)

    def _compute_discriminator_loss(
        self, y, batch_noise, batch_probs, res_probs, regressor
    ):
        y, age = y
        y = tf.reshape(y, shape=[-1])
        y = tf.one_hot(y, depth=self.num_groups) > 0

        age_loss = tf.square(age - regressor)
        mae = tf.abs(age * 100.0 - regressor * 100.0)

        # batch_noise should be as small
        noise_size = tf.reduce_sum(batch_noise * batch_noise, axis=-1)

        # cross entropy
        batch_loss = self.discriminator_loss(y, batch_probs)

        # we want to min KL divergence
        # res_probs += 1e-7  # add small constant to avoid div by 0
        uniform_p = tf.ones_like(res_probs) * (
            1.0 / tf.cast(self.num_groups, dtype=tf.float32)
        )
        H_pq = tf.reduce_sum(uniform_p * tf.math.log(1 / res_probs), axis=-1)
        H_p = tf.reduce_sum(uniform_p * tf.math.log(1 / uniform_p), axis=-1)
        res_loss = H_pq - H_p

        return (
            tf.reduce_mean(noise_size),
            tf.reduce_mean(batch_loss) * 0.01,
            tf.reduce_mean(res_loss),
            tf.reduce_mean(age_loss),
            tf.reduce_mean(mae),
        )

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        return self._encode(inputs), y

    def _process_input(self, inputs):
        asv_embeddings, batch_indices, asv_indices, asv_counts, taxonomy_counts = inputs

        batch_indices = tf.cast(batch_indices, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)

        asv_embeddings = self.sample_embeddings(
            asv_embeddings, batch_indices, asv_counts, asv_indices
        )

        taxonomy_counts = tf.cast(taxonomy_counts, dtype=tf.float32)
        total_counts = tf.reduce_sum(taxonomy_counts, axis=1, keepdims=True)
        taxonomy_counts = taxonomy_counts / total_counts
        return asv_embeddings, taxonomy_counts

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data

        with tf.GradientTape() as tape:
            (
                batch_noise,
                batch_probs,
                res_probs,
                encoder_input,
                decoder_output,
                regressor,
            ) = self(inputs, training=True)
            ae_loss = self._reconstruction_loss(encoder_input, decoder_output)
            noise_loss, batch_loss, res_loss, age_loss, mae = (
                self._compute_discriminator_loss(
                    y, batch_noise, batch_probs, res_probs, regressor
                )
            )
            loss = ae_loss + noise_loss + batch_loss + res_loss + age_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.asv_rec_tracker.update_state(ae_loss)
        self.batch_noise_Tracker.update_state(noise_loss)
        self.batch_class_tracker.update_state(batch_loss)
        self.res_class_tracker.update_state(res_loss)
        self.age_tracker.update_state(mae)
        return {
            "loss": self.loss_tracker.result(),
            "ae_loss": self.asv_rec_tracker.result(),
            "batch_noise": self.batch_noise_Tracker.result(),
            "batch_class": self.batch_class_tracker.result(),
            "res_class": self.res_class_tracker.result(),
            "age_loss": self.age_tracker.result(),
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
        (
            batch_noise,
            batch_probs,
            res_probs,
            encoder_input,
            decoder_output,
            regressor,
        ) = self(inputs, training=False)
        ae_loss = self._reconstruction_loss(encoder_input, decoder_output)
        noise_loss, batch_loss, res_loss, age_loss, mae = (
            self._compute_discriminator_loss(
                y, batch_noise, batch_probs, res_probs, regressor
            )
        )
        loss = ae_loss + noise_loss + batch_loss + res_loss + age_loss

        self.loss_tracker.update_state(loss)
        self.asv_rec_tracker.update_state(ae_loss)
        self.batch_noise_Tracker.update_state(noise_loss)
        self.batch_class_tracker.update_state(batch_loss)
        self.res_class_tracker.update_state(res_loss)
        self.age_tracker.update_state(mae)
        return {
            "loss": self.loss_tracker.result(),
            "ae_loss": self.asv_rec_tracker.result(),
            "batch_noise": self.batch_noise_Tracker.result(),
            "batch_class": self.batch_class_tracker.result(),
            "res_class": self.res_class_tracker.result(),
            "age_loss": self.age_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def batch_embeddings(self, asv_embeddings, batch_indicies, counts, asv_indices):
        emb_dim = tf.shape(asv_embeddings)[-1]
        batch_indicies = tf.cast(batch_indicies, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)

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
        return batch_embeddings, counts

    def sample_embeddings(self, asv_embeddings, batch_indicies, counts, asv_indices):
        batched_embeddigns, batch_counts = self.batch_embeddings(
            asv_embeddings, batch_indicies, counts, asv_indices
        )
        asv_mask = tf.cast(batch_counts > 0, dtype=tf.float32)
        batched_embeddigns = batched_embeddigns * asv_mask
        sample_embeddings = tf.reduce_sum(batched_embeddigns, axis=1) / tf.reduce_sum(
            asv_mask, axis=1
        )
        return sample_embeddings

    def call(
        self, inputs, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable

        if len(inputs) > 2:
            asv_embeddings, batch_indices, asv_indices, asv_counts, taxonomy_counts = (
                inputs
            )

            batch_indices = tf.cast(batch_indices, dtype=tf.int32)
            asv_indices = tf.cast(asv_indices, dtype=tf.int32)

            asv_embeddings = self.sample_embeddings(
                asv_embeddings, batch_indices, asv_counts, asv_indices
            )

            # compute relative abundance
            taxonomy_counts = tf.cast(taxonomy_counts, dtype=tf.float32)
            total_counts = tf.reduce_sum(taxonomy_counts, axis=1, keepdims=True)
            taxonomy_counts = taxonomy_counts / total_counts
        else:
            asv_embeddings, taxonomy_counts = inputs

        encoder_output, encoder_input = self.encoder([asv_embeddings, taxonomy_counts])
        discriminator_out = self.discriminator_in(encoder_output)
        batch_noise = tf.reduce_mean(discriminator_out, axis=1)
        encoder_residual = encoder_output - discriminator_out
        encode_embedding = tf.reduce_mean(encoder_residual, axis=1)

        decoder_output = self.decoder(encoder_residual, training=training)
        batch_probs = self.batch_classifier(batch_noise)
        res_probs = self.batch_classifier(encode_embedding)
        regressor = self.regressor(encode_embedding)

        print("Triplet encoder exit...")
        return (
            batch_noise,
            batch_probs,
            res_probs,
            encoder_input,
            decoder_output,
            regressor,
        )

    def _encode(self, inputs):
        if len(inputs) > 2:
            asv_embeddings, batch_indices, asv_indices, asv_counts, taxonomy_counts = (
                inputs
            )

            batch_indices = tf.cast(batch_indices, dtype=tf.int32)
            asv_indices = tf.cast(asv_indices, dtype=tf.int32)

            asv_embeddings = self.sample_embeddings(
                asv_embeddings, batch_indices, asv_counts, asv_indices
            )

            # compute relative abundance
            taxonomy_counts = tf.cast(taxonomy_counts, dtype=tf.float32)
            total_counts = tf.reduce_sum(taxonomy_counts, axis=1, keepdims=True)
            taxonomy_counts = taxonomy_counts / total_counts
        else:
            asv_embeddings, taxonomy_counts = inputs

        encoder_output, _ = self.encoder([asv_embeddings, taxonomy_counts])
        discriminator_out = self.discriminator_in(encoder_output)
        encoder_residual = encoder_output - discriminator_out
        return tf.reduce_mean(encoder_residual, axis=1)

    def get_config(self):
        config = super(TripletEncoder, self).get_config()
        config.update({"build_input_shape": self.get_build_config()})
        return config

    @classmethod
    def from_config(cls, config):
        print("Reconstructing ASVEncoder...")

        print("Constructing UnifracDenoser from config")
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls()
        model.build(input_shape)
        return model
