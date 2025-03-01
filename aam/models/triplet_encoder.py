from __future__ import annotations

from typing import Union

import tensorflow as tf
import tensorflow_addons as tfa

from aam.callbacks import LAMBLRScheduler

# from aam.data_handlers.generator_dataset import batch_embeddings
from aam.losses import _pairwise_distances, categorical_triplet_loss
from aam.models.unifrac_encoder import UnifracEncoder
from aam.models.utils import cos_decay_with_warmup, sort_using_counts
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler


def _ff_block(output_dim, use_bias=True, dropout_rate=None):
    block = [
        tf.keras.layers.Dense(
            output_dim,
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.HeUniform(),
        ),
        tf.keras.layers.LayerNormalization(dtype=tf.float32),
        tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x)),
        tf.keras.layers.Dropout(dropout_rate),
    ]
    return block


@tf.keras.saving.register_keras_serializable(package="AutoEncoder")
class AutoEncoder(tf.keras.Model):
    def __init__(self, dropout_rate, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        if self.built:
            print("AutoEncoder is already built")
            return

        asv_embeddings, taxonomy_counts = input_shape

        self.asv_encoder = tf.keras.Sequential(
            _ff_block(1024, dropout_rate=self.dropout_rate)
        )
        self.taxonomy_encoder = tf.keras.Sequential(
            _ff_block(1024, dropout_rate=self.dropout_rate)
        )
        self.encoder = tf.keras.Sequential(
            _ff_block(512, dropout_rate=self.dropout_rate)
            + _ff_block(512, dropout_rate=self.dropout_rate)
        )

        self.intermediate_ff = tf.keras.layers.Dense(512, use_bias=True)
        self.encoder_out = tf.keras.layers.Dense(512, use_bias=True)

        super(AutoEncoder, self).build(input_shape)

    def call(
        self,
        inputs,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable

        asv_embeddings, taxonomy_counts = inputs

        asv_embeddings = self.asv_encoder(asv_embeddings, training=training)
        taxonomy_embeddings = self.taxonomy_encoder(taxonomy_counts, training=training)

        intermediate_embeddings = self.encoder(asv_embeddings + taxonomy_embeddings)
        intermediate_embeddings = self.intermediate_ff(intermediate_embeddings)
        encoder_output = self.encoder_out(intermediate_embeddings)

        print("Triplet AutoEncoder exit...")
        return encoder_output, intermediate_embeddings

    def get_config(self):
        config = super(AutoEncoder, self).get_config()
        config.update({"dropout_rate": self.dropout_rate})
        return config


@tf.keras.saving.register_keras_serializable(package="TripletEncoder")
class TripletEncoder(tf.keras.Model):
    def __init__(self, dropout_rate: float = 0.3, **kwargs):
        super(TripletEncoder, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.asv_loss = _pairwise_distances
        self.asv_rec_tracker = tf.keras.metrics.Mean(name="asv_rec_loss")
        self.tax_rec_tracker = tf.keras.metrics.Mean(name="tax_rec_loss")

        self.triplet_loss = categorical_triplet_loss
        self.ortho_tracker = tf.keras.metrics.Mean(name="ortho_loss")
        self.discriminator_loss = tf.keras.losses.CategoricalCrossentropy(
            reduction="none"
        )
        self.discriminator_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        self.num_groups = 64

        self.encoder = AutoEncoder(dropout_rate=self.dropout_rate, name="auto_encoder")
        self.discriminator = tf.keras.Sequential(
            _ff_block(256, dropout_rate=self.dropout_rate)
            + _ff_block(256, dropout_rate=self.dropout_rate)
            + [
                tf.keras.layers.Dense(
                    64,
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    activation="softmax",
                )
            ]
        )

        self.ae_opt = tf.keras.optimizers.AdamW(
            cos_decay_with_warmup(0.0003, 0, 1000000),
            weight_decay=1e-5,
        )
        self.discriminator_opt = tf.keras.optimizers.AdamW(
            cos_decay_with_warmup(0.0003, 0, 1000000),
            weight_decay=1e-5,
        )

    def build(self, input_shape):
        if self.built:
            print("TripletEncoder is already built")
            return

        asv_embeddings, batch_indices, asv_indices, asv_counts, taxonomy_counts = (
            input_shape
        )
        self.encoder.build([asv_embeddings, taxonomy_counts])
        self.discriminator.build([None, 512])

        self.decoder = tf.keras.Sequential(
            _ff_block(1024, dropout_rate=self.dropout_rate)
            + _ff_block(1024, dropout_rate=self.dropout_rate)
        )
        self.decoder.build([None, 512])

        self.asv_output = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    asv_embeddings[-1],
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                )
            ],
        )
        self.asv_output.build([None, 1024])

        self.taxonomy_output = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    taxonomy_counts[-1],
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                )
            ],
        )
        self.taxonomy_output.build([None, 1024])

        super(TripletEncoder, self).build(input_shape)

    def _compute_loss(
        self,
        y,
        inputs,
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        y, sample_weights = y

        y = tf.squeeze(y, axis=-1)
        asv_embeddings, taxonomy_counts = inputs
        asv_output, tax_output = outputs

        # asv reconstruction
        asv_loss = tf.sqrt(
            tf.reduce_sum(tf.square(asv_embeddings - asv_output), axis=-1)
        )
        asv_loss = tf.reduce_mean(asv_loss * sample_weights)

        # taxonomy reconstruction
        tax_loss = tf.abs(tax_output - taxonomy_counts)
        tax_loss = tf.reduce_mean(tax_loss, axis=-1)
        tax_loss = tf.reduce_mean(tax_loss * sample_weights)

        loss = asv_loss + tax_loss
        return loss, asv_loss, tax_loss

    def _compute_discriminator_loss(self, y, outputs):
        y, sample_weights = y

        y = tf.reshape(y, shape=[-1])
        intermediate_embeddings, y_pred = outputs

        # orthogonal penalty
        groups, _ = tf.unique(y)
        num_groups = tf.shape(groups)[0]
        _, ortho_loss = self.triplet_loss(intermediate_embeddings, num_groups)
        ortho_loss = tf.reduce_mean(ortho_loss * sample_weights)

        y = tf.one_hot(y, depth=64)
        discrim_loss = self.discriminator_loss(y, y_pred)
        discrim_loss = tf.reduce_mean(discrim_loss * sample_weights)

        loss = discrim_loss + ortho_loss
        return loss, discrim_loss, ortho_loss

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        encoder_output, intermediate_embeddings = self.call(inputs, training=False)
        return encoder_output, y

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
        inputs = self._process_input(inputs)

        with tf.GradientTape() as tape:
            _, intermediate_embeddings = self(inputs, training=True)
            outputs = self.discriminator(intermediate_embeddings, training=True)
            loss, disc_loss, ortho_loss = self._compute_discriminator_loss(
                y, (intermediate_embeddings, outputs)
            )
        gradients = tape.gradient(loss, self.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(gradients, self.trainable_variables))

        with tf.GradientTape() as tape:
            encoder_output, _ = self(inputs, training=True)
            decoded_emddings = self.decoder(encoder_output, training=True)
            asv_output = self.asv_output(decoded_emddings, training=True)
            tax_output = self.taxonomy_output(decoded_emddings, training=True)
            ae_loss, asv_loss, tax_loss = self._compute_loss(
                y, inputs, (asv_output, tax_output)
            )
        gradients = tape.gradient(ae_loss, self.trainable_variables)
        self.ae_opt.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(ae_loss + loss)
        self.ortho_tracker.update_state(ortho_loss)
        self.asv_rec_tracker.update_state(asv_loss)
        self.tax_rec_tracker.update_state(tax_loss)
        self.discriminator_tracker.update_state(disc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "ortho_loss": self.ortho_tracker.result(),
            "asv_loss": self.asv_rec_tracker.result(),
            "tax_loss": self.tax_rec_tracker.result(),
            "discrim_loss": self.discriminator_tracker.result(),
            "learning_rate": self.ae_opt.learning_rate,
        }

    def test_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        inputs = self._process_input(inputs)

        encoder_output, intermediate_embeddings = self(inputs, training=False)

        # decoder
        decoded_embeddings = self.decoder(encoder_output, training=False)
        asv_output = self.asv_output(decoded_embeddings, training=False)
        tax_output = self.taxonomy_output(decoded_embeddings, training=False)
        ae_loss, asv_loss, tax_loss = self._compute_loss(
            y, inputs, (asv_output, tax_output)
        )

        # discriminator
        outputs = self.discriminator(intermediate_embeddings, training=False)
        loss, disc_loss, ortho_loss = self._compute_discriminator_loss(
            y, (intermediate_embeddings, outputs)
        )

        self.loss_tracker.update_state(ae_loss + loss)
        self.ortho_tracker.update_state(ortho_loss)
        self.asv_rec_tracker.update_state(asv_loss)
        self.tax_rec_tracker.update_state(tax_loss)
        self.discriminator_tracker.update_state(disc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "ortho_loss": self.ortho_tracker.result(),
            "asv_loss": self.asv_rec_tracker.result(),
            "tax_loss": self.tax_rec_tracker.result(),
            "discrim_loss": self.discriminator_tracker.result(),
            "learning_rate": self.ae_opt.learning_rate,
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

        encoder_output, intermediate_embeddings = self.encoder(
            [asv_embeddings, taxonomy_counts], training=training
        )

        print("Triplet encoder exit...")
        return encoder_output, intermediate_embeddings

    def get_config(self):
        config = super(TripletEncoder, self).get_config()
        config.update(
            {
                "dropout_rate": self.dropout_rate,
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

        model = cls(**config)
        model.build(input_shape)
        return model
