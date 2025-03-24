from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.losses import _pairwise_distances, global_orthogonal_regulization
from aam.models.asv_dense_count_encoder import ASVDenseCountEncoder
from aam.models.convolution_block import ConvolutionBlock


class DenseBlock2(tf.keras.layers.Layer):
    def __init__(self, pool=0, **kwargs):
        super(DenseBlock2, self).__init__(**kwargs)
        self.pool = pool

    def build(self, input_shape):
        units = input_shape[-1]
        self.dense_inner = tf.keras.layers.Dense(units, activation="gelu")

        if self.pool < 0:
            self.dense_outer = tf.keras.layers.Dense(units // 2)
            self.res_pool = tf.keras.layers.Dense(units // 2)
        elif self.pool > 0:
            self.dense_outer = tf.keras.layers.Dense(units * 2)
            self.res_pool = tf.keras.layers.Dense(units * 2)
        else:
            self.dense_outer = tf.keras.layers.Dense(units)

        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, inputs, training=False):
        output = self.dense_inner(inputs)
        output = self.dense_outer(output)

        # residual step
        if self.pool:
            inputs = self.res_pool(inputs)
        output = inputs + self._rezero * output
        return output

    def get_config(self):
        return super().get_config().update({"pool": self.pool})


@tf.keras.saving.register_keras_serializable(package="TripletEncoderV4")
class TripletEncoderV4(tf.keras.Model):
    def __init__(
        self,
        num_groups,
        unifrac_model,
        num_noise_layers=6,
        compress_factor=3,
        num_filters=8,
        kernel_size=3,
        pool_size=2,
        **kwargs,
    ):
        super(TripletEncoderV4, self).__init__(**kwargs)
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
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size

    def build(self, input_shape):
        if self.built:
            print("TripletEncoderV4 is already built")
            return

        asv_embeddings, batch_indices, asv_indices, asv_counts, dense_counts = (
            input_shape
        )

        # # build Encoder
        # encoder_layers = [tf.keras.layers.Input([asv_embeddings[-1], 1])]
        # emb_dim = asv_embeddings[-1]
        # for _ in range(self.compress_factor):
        #     encoder_layers += [
        #         ConvolutionBlock(self.num_filters, self.kernel_size, pool_size=0),
        #         ConvolutionBlock(
        #             self.num_filters, self.kernel_size, pool_size=self.pool_size
        #         ),
        #     ]
        #     emb_dim /= self.pool_size
        # emb_dim = int(emb_dim)
        # self.encoder = tf.keras.Sequential(
        #     encoder_layers
        #     + [
        #         tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1)),
        #         tf.keras.layers.Dense(emb_dim),
        #     ],
        #     name="encoder",
        # )

        # discriminator_layers = [tf.keras.layers.Input([emb_dim, 1])]
        # for _ in range(self.num_noise_layers):
        #     discriminator_layers += [
        #         ConvolutionBlock(self.num_filters, self.kernel_size, pool_size=0)
        #     ]
        # self.discriminator = tf.keras.Sequential(
        #     discriminator_layers
        #     + [
        #         tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1)),
        #         tf.keras.layers.Dense(emb_dim),
        #     ],
        #     name="discriminator",
        # )

        # build Encoder
        compression_size = asv_embeddings[-1] // (2**self.compress_factor)

        encoder_layers = []
        for _ in range(self.compress_factor):
            encoder_layers += [
                DenseBlock2(pool=-1),
            ]
        self.encoder = tf.keras.Sequential(
            encoder_layers + [tf.keras.layers.Dense(compression_size)],
            name="encoder",
        )

        dense_size = dense_counts[-1]
        discriminator_layers = [tf.keras.layers.Input([dense_size, 1])]
        i = 0
        while dense_size > compression_size:
            discriminator_layers += [
                ConvolutionBlock(
                    self.num_filters,
                    self.kernel_size,
                    pool_size=0,
                ),
                ConvolutionBlock(
                    self.num_filters,
                    self.kernel_size,
                    pool_size=self.pool_size,
                    use_max_pool=False,
                ),
            ]
            dense_size /= self.pool_size
            i += 1
        print(f"{i} dense conv layers")

        self.discriminator = tf.keras.Sequential(
            discriminator_layers
            + [
                tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1)),
                tf.keras.layers.Dense(compression_size),
            ],
            name="discriminator",
        )

        # build Decoder
        decoder_layers = []
        for _ in range(self.compress_factor):
            decoder_layers += [
                DenseBlock2(pool=1),
            ]
        self.decoder = tf.keras.Sequential(
            decoder_layers + [tf.keras.layers.Dense(asv_embeddings[-1])],
            name="decoder",
        )

        self.batch_classifier = tf.keras.layers.Dense(
            self.num_groups,
            use_bias=True,
            activation="softmax",
            name="batch_classifier",
        )

        super(TripletEncoderV4, self).build(input_shape)

    def _reconstruction_loss(self, encoder_input, decoder_output):
        square_difference = tf.reduce_sum(
            tf.square(encoder_input - decoder_output), axis=-1
        )
        return tf.reduce_mean(square_difference)

    def _compute_discriminator_loss(
        self,
        y,
        batch_noise,
        batch_probs,
        res_probs,  # , regressor
    ):
        y, sample_weights = y

        # y, age = y
        y = tf.reshape(y, shape=[-1])
        y = tf.one_hot(y, depth=self.num_groups) > 0

        # batch_noise should be as small
        noise_size = tf.reduce_sum(tf.abs(batch_noise), axis=-1)

        # cross entropy
        batch_loss = self.discriminator_loss(y, batch_probs)

        # we want to min KL divergence
        uniform = tf.ones_like(res_probs) * (
            1.0 / tf.cast(self.num_groups, dtype=tf.float32)
        )
        p = uniform
        q = res_probs
        log_pq = tf.math.log(p) - tf.math.log(q)
        kl = tf.reduce_sum(p * log_pq, axis=-1)

        return (
            tf.reduce_mean(noise_size),
            tf.reduce_mean(batch_loss) * 0.01,
            tf.reduce_mean(kl),
        )

    def _compute_batch_noise(self, residual, batch_noise):
        output_norm = tf.norm(residual, axis=-1, keepdims=True)
        batch_norn = tf.norm(batch_noise, axis=-1, keepdims=True)
        mask = tf.cast(batch_norn >= 0.35 * output_norm, dtype=tf.float32)
        loss = tf.reduce_sum(batch_norn * mask)
        return tf.math.divide_no_nan(loss, tf.reduce_sum(mask)) * 10.0

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

        with tf.GradientTape() as tape:
            (
                batch_noise,
                batch_probs,
                res_probs,
                encoder_input,
                decoder_output,
                encoder_residual,
            ) = self(inputs, return_training_output=True, training=True)
            ae_loss = self._reconstruction_loss(encoder_input, decoder_output)
            noise_loss, batch_loss, res_loss = self._compute_discriminator_loss(
                y, batch_noise, batch_probs, res_probs
            )
            batch_noise_loss = self._compute_batch_noise(encoder_residual, batch_noise)
            loss = ae_loss + batch_loss + res_loss + batch_noise_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

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
            encoder_residual,
        ) = self(inputs, return_training_output=True, training=False)
        ae_loss = self._reconstruction_loss(encoder_input, decoder_output)
        noise_loss, batch_loss, res_loss = self._compute_discriminator_loss(
            y, batch_noise, batch_probs, res_probs
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
            "learning_rate": self.optimizer.learning_rate,
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
        asv_embeddings, batch_indices, asv_indices, asv_counts, dense_counts = inputs
        encoder_input = self.unifrac_model(inputs, training=False)
        encoder_output = self.encoder(encoder_input)

        dense_counts = self._log1p_relative_abundance(dense_counts)
        batch_noise = self.discriminator(dense_counts)

        # batch_noise = self.discriminator(encoder_output)
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
        config = super(TripletEncoderV4, self).get_config()
        config.update(
            {
                "num_groups": self.num_groups,
                "unifrac_model": tf.keras.saving.serialize_keras_object(
                    self.unifrac_model
                ),
                "num_noise_layers": self.num_noise_layers,
                "compress_factor": self.compress_factor,
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
