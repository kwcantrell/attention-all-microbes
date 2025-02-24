from __future__ import annotations

from typing import Union

import tensorflow as tf

# from aam.data_handlers.generator_dataset import batch_embeddings
from aam.losses import PairwiseLoss, triplet_loss
from aam.models.unifrac_encoder_v2 import UnifracEncoderV2
from aam.models.utils import sort_using_counts, to_batch
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler


@tf.keras.saving.register_keras_serializable(package="UnifracDenoiserV2")
class UnifracDenoiserV2(tf.keras.Model):
    def __init__(
        self,
        dropout_rate: float = 0.0,
        embedding_dim: int = 32,
        asv_encoder=None,
        **kwargs,
    ):
        super(UnifracDenoiserV2, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim

        if asv_encoder is None:
            raise Exception("UnifracDeniser is missing ASVEncoder")
        self.asv_encoder = asv_encoder
        self.asv_encoder.trainable = False

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pairwise_loss = PairwiseLoss()
        self.triplet_loss = triplet_loss
        self.unifrac_tracker = tf.keras.metrics.Mean(name="unifrac_loss")
        self.denoise_tracker = tf.keras.metrics.Mean(name="denoised_loss")

        def _ff_block():
            block = [
                tf.keras.layers.Dense(
                    self.embedding_dim,
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                ),
                tf.keras.layers.LayerNormalization(dtype=tf.float32),
                tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x)),
                tf.keras.layers.Dropout(self.dropout_rate),
            ]

            return block

        self.sample_ff = tf.keras.Sequential(
            [tf.keras.layers.LayerNormalization(dtype=tf.float32)] + _ff_block(),
            name="sample_ff",
        )
        self.unifrac_ff = tf.keras.Sequential(
            _ff_block() + [tf.keras.layers.Dense(32)], name="unifrac_ff"
        )
        self.denoise_ff = tf.keras.Sequential(
            _ff_block() + [tf.keras.layers.Dense(32)], name="denoise_ff"
        )
        self.output_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)

    def _compute_unifrac_loss(self, unifrac_distances, unifrac_embeddings):
        shape = tf.shape(unifrac_distances)
        batch_dim = shape[0]
        group_dim = shape[-1]
        groups = batch_dim // group_dim
        unifrac_distances = tf.reshape(
            unifrac_distances, shape=[groups, group_dim, group_dim]
        )
        unifrac_embeddings = tf.reshape(
            unifrac_embeddings, shape=[groups, group_dim, self.embedding_dim]
        )

        def _unifrac_loss(inputs):
            uni_dist, uni_emb = inputs
            return tf.reduce_mean(self.pairwise_loss(uni_dist, uni_emb))

        losses = tf.map_fn(
            _unifrac_loss,
            [unifrac_distances, unifrac_embeddings],
            fn_output_signature=tf.float32,
        )

        return tf.reduce_mean(losses)

    def _compute_denoise_loss(self, denoised_embeddings):
        denoise_loss = self.triplet_loss(denoised_embeddings)
        return tf.reduce_mean(denoise_loss)

    def _compute_loss(
        self,
        unifrac_distances: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        unifrac_embeddings, denoised_embeddings = outputs

        unifrac_loss = self._compute_unifrac_loss(unifrac_distances, unifrac_embeddings)
        denoise_loss = self._compute_denoise_loss(denoised_embeddings)

        loss = unifrac_loss + denoise_loss
        return loss, unifrac_loss, denoise_loss

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        unifrac_embeddings, denoise_unifrac_embeddings = self.call(
            inputs, training=False
        )
        return denoise_unifrac_embeddings, y

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        y_target, encoder_target = y

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, unifrac_loss, denoise_loss = self._compute_loss(
                encoder_target, outputs
            )
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(unifrac_loss + denoise_loss)
        self.unifrac_tracker.update_state(unifrac_loss)
        self.denoise_tracker.update_state(denoise_loss)

        return {
            "loss": self.loss_tracker.result(),
            "unifrac_loss": self.unifrac_tracker.result(),
            "denoise_loss": self.denoise_tracker.result(),
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
        y_target, encoder_target = y

        outputs = self(inputs, training=False)
        loss, unifrac_loss, denoise_loss = self._compute_loss(encoder_target, outputs)

        self.loss_tracker.update_state(loss)
        self.unifrac_tracker.update_state(unifrac_loss)
        self.denoise_tracker.update_state(denoise_loss)
        return {
            "loss": self.loss_tracker.result(),
            "unifrac_loss": self.unifrac_tracker.result(),
            "denoise_loss": self.denoise_tracker.result(),
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
        sample_embeddings = tf.reduce_sum(batched_embeddigns, axis=1) / tf.reduce_sum(
            asv_mask, axis=1
        )
        return sample_embeddings

    def call(
        self, inputs, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        tokens, batch_indices, asv_indices, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        batch_indices = tf.cast(batch_indices, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)
        asv_embeddings = self.asv_encoder.asv_embeddings(tokens)
        sample_embeddings = self.sample_embeddings(
            asv_embeddings, batch_indices, counts, asv_indices
        )

        sample_embeddings = self.sample_ff(sample_embeddings, training=training)
        unifrac_embeddings = self.unifrac_ff(sample_embeddings, training=training)
        denoised_embeddings = self.denoise_ff(unifrac_embeddings, training=training)
        return (
            self.output_activation(unifrac_embeddings),
            self.output_activation(denoised_embeddings),
        )

    def get_config(self):
        config = super(UnifracDenoiserV2, self).get_config()
        config.update(
            {
                "dropout_rate": self.dropout_rate,
                "embedding_dim": self.embedding_dim,
                "asv_encoder": tf.keras.saving.serialize_keras_object(self.asv_encoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        print("Reconstructing ASVEncoder...")
        asv_encoder = tf.keras.saving.deserialize_keras_object(config["asv_encoder"])
        asv_encoder.trainable = False
        config["asv_encoder"] = asv_encoder

        model = cls(**config)
        return model
