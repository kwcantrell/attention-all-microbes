from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.losses import PairwiseLoss
from aam.models.asv_dense_count_encoder import ASVDenseCountEncoder
from aam.models.regressor_v2 import DenseBlock


@tf.keras.saving.register_keras_serializable(package="UnifracEncoderV4")
class UnifracEncoderV4(tf.keras.Model):
    def __init__(self, num_encoder_layers=4, **kwargs):
        super(UnifracEncoderV4, self).__init__(**kwargs)
        self.unifrac_loss = PairwiseLoss()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.num_encoder_layers = num_encoder_layers

    def build(self, input_shape):
        if self.built:
            print("UnifracEncoderV4 is already built")
            return

        asv_embeddings, batch_indices, asv_indices, asv_counts, dense_counts = (
            input_shape
        )
        self.asv_encoder = ASVDenseCountEncoder(name="asv_encoder")

        encoder_layers = []
        for _ in range(self.num_encoder_layers):
            encoder_layers += [
                DenseBlock(),
            ]
        self.encoder = tf.keras.Sequential(
            encoder_layers + [tf.keras.layers.Dense(asv_embeddings[-1])],
            name="encoder",
        )
        super(UnifracEncoderV4, self).build(input_shape)

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
            loss = self._compute_loss(y, output_embeddings)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
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
        output_embeddings = self(inputs, training=False)
        loss = self._compute_loss(y, output_embeddings)

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
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

    def _normalize_dense_counts(self, dense_counts):
        # compute relative abundance
        dense_counts = tf.cast(dense_counts, dtype=tf.float32)
        total_counts = tf.reduce_sum(dense_counts, axis=-1)
        depth = tf.reduce_max(total_counts)
        dense_counts /= depth

        # normalize counts
        dense_counts = tf.math.log1p(dense_counts)
        dense_mean = tf.reduce_mean(dense_counts, axis=1, keepdims=True)
        dense_std = tf.math.reduce_std(dense_counts, axis=1, keepdims=True)
        return (dense_counts - dense_mean) / dense_std

    def call(
        self, inputs, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        asv_embeddings, batch_indices, asv_indices, asv_counts, dense_counts = inputs

        batch_indices = tf.cast(batch_indices, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)

        asv_embeddings = self.sample_embeddings(
            asv_embeddings, batch_indices, asv_counts, asv_indices
        )
        dense_counts = self._normalize_dense_counts(dense_counts)

        encoder_input = self.asv_encoder([asv_embeddings, dense_counts])
        output_embeddings = self.encoder(encoder_input)
        print("UnifracEncoderV4 exit...")
        return output_embeddings

    def get_config(self):
        config = super(UnifracEncoderV4, self).get_config()
        config.update(
            {
                "num_encoder_layers": self.num_encoder_layers,
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        print("Constructing UnifracEncoderV4 from config")
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)
        model.build(input_shape)
        return model
