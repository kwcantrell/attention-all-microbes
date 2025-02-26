from __future__ import annotations

from typing import Union

import tensorflow as tf

# from aam.data_handlers.generator_dataset import batch_embeddings
from aam.losses import PairwiseLoss, categorical_triplet_loss
from aam.models.unifrac_encoder import UnifracEncoder
from aam.models.utils import sort_using_counts
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler


@tf.keras.saving.register_keras_serializable(package="TripletEncoder")
class TripletEncoder(tf.keras.Model):
    def __init__(
        self,
        asv_encoder=None,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super(TripletEncoder, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

        if asv_encoder is None:
            raise Exception("UnifracDeniser is missing ASVEncoder")
        self.asv_encoder = asv_encoder
        self.asv_encoder.trainable = False

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.triplet_loss = categorical_triplet_loss
        self.triplet_tracker = tf.keras.metrics.Mean(name="triplet_loss")
        self.ortho_tracker = tf.keras.metrics.Mean(name="triplet_loss")

    # def build(self, input_shape):
    #     if self.built:
    #         print("TripletEncoder is already built")
    #         return

    #     def _ff_block(output_dim, use_bias=True, dropout_rate=None):
    #         block = [
    #             tf.keras.layers.LayerNormalization(dtype=tf.float32),
    #             tf.keras.layers.Dense(
    #                 output_dim,
    #                 use_bias=use_bias,
    #                 kernel_initializer=tf.keras.initializers.HeUniform(),
    #             ),
    #             tf.keras.layers.LayerNormalization(dtype=tf.float32),
    #             tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x)),
    #         ]
    #         if dropout_rate:
    #             block.append(tf.keras.layers.Dropout(dropout_rate))
    #         return block

    #     self.sample_embedding_ff = tf.keras.Sequential(_ff_block(32, dropout_rate=0.5))
    #     self.tax_count_ff = tf.keras.Sequential(_ff_block(32, dropout_rate=0.5))
    #     self.out_ff = tf.keras.layers.Dense(
    #         32, kernel_initializer=tf.keras.initializers.HeUniform(), dtype=tf.float32
    #     )
    #     self.output_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)

    #     super(TripletEncoder, self).build(input_shape)
    def build(self, input_shape):
        if self.built:
            print("TripletEncoder is already built")
            return

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

        self.sample_embedding_ff = tf.keras.Sequential(_ff_block(128, dropout_rate=0.0))
        self.tax_count_ff = tf.keras.Sequential(_ff_block(128, dropout_rate=0.0))
        self.out_ff = tf.keras.layers.Dense(32, use_bias=True)
        self.output_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)

        super(TripletEncoder, self).build(input_shape)

    def _compute_loss(
        self,
        y,
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        triplet_embeddings = outputs

        y = tf.squeeze(y, axis=-1)
        groups, _ = tf.unique(y)
        num_groups = tf.shape(groups)[0]
        triplet_loss, ortho_loss = self.triplet_loss(triplet_embeddings, num_groups)
        triplet_loss = tf.reduce_mean(triplet_loss)
        ortho_loss = tf.reduce_mean(ortho_loss)
        loss = triplet_loss + ortho_loss
        return loss, triplet_loss, ortho_loss

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        triplet_embeddings = self.call(inputs, training=False)
        return triplet_embeddings, y

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, triplet_loss, ortho_loss = self._compute_loss(y, outputs)
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.triplet_tracker.update_state(triplet_loss)
        self.ortho_tracker.update_state(ortho_loss)
        return {
            "loss": self.loss_tracker.result(),
            "triplet_loss": self.triplet_tracker.result(),
            "ortho_loss": self.ortho_tracker.result(),
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

        outputs = self(inputs, training=False)
        loss, triplet_loss, ortho_loss = self._compute_loss(y, outputs)

        self.loss_tracker.update_state(loss)
        self.triplet_tracker.update_state(triplet_loss)
        self.ortho_tracker.update_state(ortho_loss)
        return {
            "loss": self.loss_tracker.result(),
            "triplet_loss": self.triplet_tracker.result(),
            "ortho_loss": self.ortho_tracker.result(),
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

    # def call(
    #     self,
    #     inputs,
    #     training: bool = False,
    # ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    #     training = training and self.trainable

    #     inputs, taxonomy_counts = inputs[:4], inputs[4]
    #     tokens, batch_indices, asv_indices, counts = inputs
    #     tokens = tf.cast(tokens, dtype=tf.int32)
    #     batch_indices = tf.cast(batch_indices, dtype=tf.int32)
    #     asv_indices = tf.cast(asv_indices, dtype=tf.int32)

    #     asv_embeddings = self.asv_encoder.asv_embeddings(tokens)
    #     sample_embeddings = self.sample_embeddings(
    #         asv_embeddings, batch_indices, counts, asv_indices
    #     )

    #     sample_embeddings = self.sample_embedding_ff(
    #         sample_embeddings, training=training
    #     )

    #     # compute relative abundance
    #     taxonomy_counts = tf.cast(taxonomy_counts, dtype=tf.float32)
    #     total_counts = tf.reduce_sum(taxonomy_counts, axis=1, keepdims=True)
    #     taxonomy_counts = taxonomy_counts / total_counts
    #     taxonomy_counts = self.tax_count_ff(taxonomy_counts, training=training)

    #     sample_embeddings = (sample_embeddings + taxonomy_counts) / 2.0
    #     output = self.out_ff(sample_embeddings)
    #     print("Triplet encoder exit...")
    #     return self.output_activation(output)
    def call(
        self,
        inputs,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable

        inputs, taxonomy_counts = inputs[:4], inputs[4]
        tokens, batch_indices, asv_indices, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        batch_indices = tf.cast(batch_indices, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)

        asv_embeddings = self.asv_encoder.asv_embeddings(tokens)
        sample_embeddings = self.sample_embeddings(
            asv_embeddings, batch_indices, counts, asv_indices
        )
        sample_embeddings = self.sample_embedding_ff(
            sample_embeddings, training=training
        )

        # compute relative abundance
        taxonomy_counts = tf.cast(taxonomy_counts, dtype=tf.float32)
        total_counts = tf.reduce_sum(taxonomy_counts, axis=1, keepdims=True)
        taxonomy_counts = taxonomy_counts / total_counts
        taxonomy_embeddings = self.tax_count_ff(taxonomy_counts, training=training)

        output_embedding = sample_embeddings + taxonomy_embeddings
        # output_embedding = self.out_ff(output_embedding)
        print("Triplet encoder exit...")
        return self.output_activation(self.out_ff(output_embedding))

    def get_config(self):
        config = super(TripletEncoder, self).get_config()
        config.update(
            {
                "dropout_rate": self.dropout_rate,
                "build_input_shape": self.get_build_config(),
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

        print("Constructing UnifracDenoser from config")
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)
        model.build(input_shape)
        return model
