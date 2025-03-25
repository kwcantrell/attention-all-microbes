from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.losses import _pairwise_distances, global_orthogonal_regulization
from aam.models.conv_feedforward import ConvFeedForward
from aam.models.convolution_block import ConvolutionBlock
from aam.models.feedforward import FeedForward


@tf.keras.saving.register_keras_serializable(package="TripletEncoder")
class TripletEncoder(tf.keras.Model):
    def __init__(
        self,
        num_groups,
        unifrac_model,
        num_noise_layers=6,
        compress_factor=1,
        blocks_per_compression=6,
        num_filters=64,
        kernel_size=3,
        pool_size=2,
        **kwargs,
    ):
        super(TripletEncoder, self).__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.asv_loss = _pairwise_distances
        self.asv_rec_tracker = tf.keras.metrics.Mean(name="asv_rec_loss")
        self.batch_noise_Tracker = tf.keras.metrics.Mean(name="batch_noise_loss")

        self.triplet_loss = global_orthogonal_regulization
        self.res_class_tracker = tf.keras.metrics.Mean(name="ortho_loss")
        self.discriminator_loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.batch_mag_tracker = tf.keras.metrics.Mean(name="discriminator_mag")
        self.batch_class_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        self.age_tracker = tf.keras.metrics.Mean(name="age_loss")
        self.num_groups = num_groups
        self.unifrac_model = unifrac_model
        self.unifrac_model.trainable = False

        self.num_noise_layers = num_noise_layers
        self.compress_factor = compress_factor
        self.blocks_per_compression = blocks_per_compression
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.group_explained = 0.35

    def build(self, input_shape):
        if self.built:
            print("TripletEncoder is already built")
            return

        encoder_layers = []
        for i in range(self.compress_factor):
            for _ in range(self.blocks_per_compression):
                encoder_layers += [ConvFeedForward(self.num_filters, self.kernel_size)]
            encoder_layers += [
                ConvFeedForward(self.num_filters, self.kernel_size),
            ]
        self.encoder = tf.keras.Sequential(encoder_layers, name="encoder")

        discriminator_layers = []
        for _ in range(self.num_noise_layers):
            discriminator_layers += [
                ConvFeedForward(self.num_filters, self.kernel_size)
            ]
        self.discriminator = tf.keras.Sequential(
            discriminator_layers, name="discriminator"
        )

        decoder_layers = []
        for _ in range(self.compress_factor):
            for _ in range(self.blocks_per_compression):
                decoder_layers += [ConvFeedForward(self.num_filters, self.kernel_size)]
            decoder_layers += [
                ConvFeedForward(self.num_filters, self.kernel_size),
            ]
        self.decoder = tf.keras.Sequential(decoder_layers, name="decoder")

        self.batch_classifier = tf.keras.Sequential(
            [
                FeedForward(outdim=self.num_groups),
                tf.keras.layers.Activation("softmax"),
            ],
            name="batch_classifier",
        )

        super(TripletEncoder, self).build(input_shape)

    def compile(self, ae_optimizer, disc_optimizer, **kwargs):
        super(TripletEncoder, self).compile(**kwargs)

        self.ae_optimizer = ae_optimizer
        self.disc_optimizer = disc_optimizer

    def _reconstruction_loss(self, encoder_input, decoder_output):
        reconstruction_loss = tf.norm(encoder_input - decoder_output, axis=-1)
        return tf.reduce_mean(reconstruction_loss)

    def _compute_discriminator_loss(self, y, batch_probs, res_probs):
        y, sample_weights = y

        # cross entropy
        y = tf.reshape(y, shape=[-1])
        y = tf.one_hot(y, depth=self.num_groups, dtype=tf.float32)
        batch_loss = self.discriminator_loss(y, batch_probs)

        # we want to min KL divergence
        # uniform = tf.ones_like(res_probs) * (
        #     1.0 / tf.cast(self.num_groups, dtype=tf.float32)
        # )
        # p = uniform
        remaining_explained = (1.0 - self.group_explained) / self.num_groups
        p = y * (self.group_explained - remaining_explained) + remaining_explained
        q = res_probs
        log_pq = tf.math.log(p) - tf.math.log(q)
        kl = tf.reduce_sum(p * log_pq, axis=-1)
        # mask = y > 0
        # kl = -1.0 * tf.math.log(res_probs[mask] + 1e-7)

        return tf.reduce_mean(batch_loss), tf.reduce_mean(kl)

    def _compute_batch_noise(self, encoder_residual, discriminator_output):
        encoder_norm = tf.norm(encoder_residual, axis=-1)
        discriminator_norn = tf.norm(discriminator_output, axis=-1)
        mask = tf.cast(
            discriminator_norn >= self.group_explained * encoder_norm, dtype=tf.float32
        )
        return tf.reduce_mean(discriminator_norn * mask)

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        return self(inputs, training=False), y

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data

        with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
            (
                batch_noise,
                batch_probs,
                res_probs,
                encoder_input,
                decoder_output,
                encoder_residual,
            ) = self(inputs, return_training_output=True, training=True)
            rec_loss = self._reconstruction_loss(encoder_input, decoder_output)
            batch_loss, res_loss = self._compute_discriminator_loss(
                y, batch_probs, res_probs
            )
            batch_noise_loss = self._compute_batch_noise(encoder_residual, batch_noise)

            ae_loss = rec_loss
            disc_loss = batch_loss + res_loss + batch_noise_loss
        disc_trainable = (
            self.discriminator.trainable_variables
            + self.batch_classifier.trainable_variables
        )
        ae_trainable = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )

        disc_gradients = disc_tape.gradient(disc_loss, disc_trainable)
        ae_gradients = ae_tape.gradient(ae_loss, ae_trainable)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, disc_trainable))
        self.ae_optimizer.apply_gradients(zip(ae_gradients, ae_trainable))

        self.loss_tracker.update_state(ae_loss + disc_loss)
        self.asv_rec_tracker.update_state(rec_loss)
        self.batch_noise_Tracker.update_state(batch_noise_loss)
        self.batch_class_tracker.update_state(batch_loss)
        self.res_class_tracker.update_state(res_loss)
        return {
            "loss": self.loss_tracker.result(),
            "ae_loss": self.asv_rec_tracker.result(),
            "batch_noise": self.batch_noise_Tracker.result(),
            "batch_class": self.batch_class_tracker.result(),
            "kl": self.res_class_tracker.result(),
            "learning_rate": self.ae_optimizer.learning_rate,
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
            encoder_residual,
        ) = self(inputs, return_training_output=True, training=False)
        ae_loss = self._reconstruction_loss(encoder_input, decoder_output)
        batch_loss, res_loss = self._compute_discriminator_loss(
            y, batch_probs, res_probs
        )
        batch_noise_loss = self._compute_batch_noise(encoder_residual, batch_noise)
        loss = ae_loss + batch_loss + res_loss + batch_noise_loss

        self.loss_tracker.update_state(loss)
        self.asv_rec_tracker.update_state(ae_loss)
        self.batch_noise_Tracker.update_state(batch_noise_loss)
        self.batch_class_tracker.update_state(batch_loss)
        self.res_class_tracker.update_state(res_loss)
        return {
            "loss": self.loss_tracker.result(),
            "ae_loss": self.asv_rec_tracker.result(),
            "batch_noise": self.batch_noise_Tracker.result(),
            "batch_class": self.batch_class_tracker.result(),
            "kl": self.res_class_tracker.result(),
            "learning_rate": self.ae_optimizer.learning_rate,
        }

    def _log1p_relative_abundance(self, dense_counts):
        # compute relative abundance
        dense_counts = tf.cast(dense_counts, dtype=tf.float32)
        total_counts = tf.reduce_sum(dense_counts, axis=-1)
        depth = tf.reduce_max(total_counts)
        dense_counts /= depth

        # normalize counts
        dense_counts = tf.math.log1p(dense_counts)
        return dense_counts

    def call(
        self, inputs, return_training_output=False, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        encoder_input = self.unifrac_model(inputs, training=False)
        encoder_output = self.encoder(encoder_input)
        batch_noise = self.discriminator(encoder_output)
        encoder_residual = encoder_output - batch_noise
        decoder_output = self.decoder(encoder_residual)

        batch_probs = self.batch_classifier(batch_noise)
        res_probs = self.batch_classifier(encoder_residual)

        print("Triplet encoder exit...")
        if not return_training_output:
            return encoder_residual

        return (
            batch_noise,
            batch_probs,
            res_probs,
            encoder_input,
            decoder_output,
            encoder_residual,
        )

    def get_config(self):
        config = super(TripletEncoder, self).get_config()
        config.update(
            {
                "num_groups": self.num_groups,
                "unifrac_model": tf.keras.saving.serialize_keras_object(
                    self.unifrac_model
                ),
                "num_noise_layers": self.num_noise_layers,
                "compress_factor": self.compress_factor,
                "blocks_per_compression": self.blocks_per_compression,
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        print("Reconstructing ASVEncoder...")

        print("Constructing UnifracDenoser from config")
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        config["unifrac_model"] = tf.keras.saving.deserialize_keras_object(
            config["unifrac_model"]
        )
        model = cls(**config)
        model.build(input_shape)
        return model
