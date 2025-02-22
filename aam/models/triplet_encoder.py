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

    def build(self, input_shape):
        if self.built:
            print("TripletEncoder is already built")
            return

        self.sample_emb_batch_norm = tf.keras.layers.BatchNormalization(
            dtype=tf.float32
        )
        self.taxon_count_batch_norm = tf.keras.layers.BatchNormalization(
            dtype=tf.float32
        )

        def _ff_block(current_dim, use_bias=True):
            return [
                tf.keras.layers.Dense(
                    current_dim,
                    use_bias=use_bias,
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                ),
                tf.keras.layers.BatchNormalization(dtype=tf.float32),
                tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x)),
                tf.keras.layers.Dropout(0.25),
            ]

        ff_layers = []
        embedding_dim = 256
        current_dim = embedding_dim
        while current_dim > 32:
            ff_layers += _ff_block(current_dim)
            ff_layers += _ff_block(current_dim)
            ff_layers += _ff_block(current_dim // 2)
            current_dim = current_dim // 2
        self.regressor = tf.keras.Sequential(ff_layers)

        self.out_emb_ff = tf.keras.layers.Dense(
            32, kernel_initializer=tf.keras.initializers.HeUniform()
        )
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
        triplet_loss = tf.reduce_mean(self.triplet_loss(triplet_embeddings, num_groups))

        loss = triplet_loss
        return loss, triplet_loss

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
            loss, triplet_loss = self._compute_loss(y, outputs)
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(triplet_loss)
        self.triplet_tracker.update_state(triplet_loss)

        return {
            "loss": self.loss_tracker.result(),
            "triplet_loss": self.triplet_tracker.result(),
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
        loss, triplet_loss = self._compute_loss(y, outputs)

        self.loss_tracker.update_state(triplet_loss)
        self.triplet_tracker.update_state(triplet_loss)
        return {
            "loss": self.loss_tracker.result(),
            "triplet_loss": self.triplet_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def call(
        self,
        inputs,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable
        if len(inputs) == 5:
            inputs, taxonomy_counts = inputs[:4], inputs[4]
            asv_embeddings, counts = self.asv_encoder.asv_embeddings(inputs)
            mask = tf.cast(counts > 0, dtype=self.compute_dtype)
            asv_embeddings = tf.cast(asv_embeddings, dtype=self.compute_dtype) * mask

            sample_embeddings = tf.reduce_sum(asv_embeddings, axis=1) / tf.reduce_sum(
                mask, axis=1
            )
            sample_embeddings = self.sample_emb_batch_norm(
                sample_embeddings, training=training
            )

            # compute relative abundance
            taxonomy_counts = tf.cast(taxonomy_counts, dtype=tf.float32)
            total_counts = tf.reduce_sum(taxonomy_counts, axis=1, keepdims=True)
            taxonomy_counts = taxonomy_counts / total_counts
            taxonomy_counts = self.taxon_count_batch_norm(
                taxonomy_counts, training=training
            )

            sample_embeddings = tf.concat([sample_embeddings, taxonomy_counts], axis=1)
            sample_embeddings = self.regressor(sample_embeddings, training=training)
            sample_embeddings = self.out_emb_ff(sample_embeddings)
            print("Triplet encoder exit...")
            return self.output_activation(sample_embeddings)
        else:
            asv_embeddings, counts = self.asv_encoder.asv_embeddings(inputs)
            mask = tf.cast(counts > 0, dtype=self.compute_dtype)
            asv_embeddings = tf.cast(asv_embeddings, dtype=self.compute_dtype) * mask

            sample_embeddings = tf.reduce_sum(asv_embeddings, axis=1) / tf.reduce_sum(
                mask, axis=1
            )
            sample_embeddings = self.regressor(sample_embeddings, training=training)
            return self.output_activation(sample_embeddings)

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
