from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.losses import PairwiseLoss
from aam.models.convolution_pooler_v2 import NonPoolingBlock, PoolingBlock
from aam.models.feedforward import FeedForward
from aam.models.transformers import TransformerEncoder
from aam.utils import create_random_mask


@tf.keras.saving.register_keras_serializable(package="RegressorV2")
class RegressorV2(tf.keras.Model):
    def __init__(
        self,
        shift,
        scale,
        base_model,
        num_filters=256,
        kernel_size=3,
        pooling_size=512,
        **kwargs,
    ):
        super(RegressorV2, self).__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_tracker = tf.keras.metrics.Mean(name="mae")
        self.block_tracker = tf.keras.metrics.Mean(name="pair")
        self.pair_loss = PairwiseLoss()
        self.block_entropy = tf.keras.losses.SparseCategoricalCrossentropy()

        self.shift = shift
        self.scale = scale
        base_model.trainable = False
        self.base_model = base_model
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size

    def build(self, input_shape):
        if self.built:
            print("RegressorV2 is already built")
            return

        self.count_extractor = tf.keras.Sequential(
            [
                PoolingBlock(self.num_filters, self.pooling_size),
            ],
            name="count_extractor",
        )
        self.block_pred = tf.keras.Sequential(
            [
                FeedForward(),
                tf.keras.layers.Dense(2, activation="softmax"),
            ],
            name="block_pred",
        )

        self.encoder = TransformerEncoder(intermediate_size=1024, name="encoder")
        self.regressor = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
                FeedForward(),
                tf.keras.layers.Dense(units=1),
            ],
            name="regressor",
        )
        super(RegressorV2, self).build(input_shape)

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        output, _, _ = self(inputs, training=False)
        y = y * self.scale + self.shift
        output = output * self.scale + self.shift
        return output, y

    def _compute_loss(self, y, output):
        output, positions, block_prob = output
        mse = tf.reduce_mean(tf.abs(y - output))
        position_loss = tf.reduce_mean(self.block_entropy(positions, block_prob))
        return mse, position_loss

    def _compute_metric(self, y, output):
        output, block, block_pred = output
        y = y * self.scale + self.shift
        output = output * self.scale + self.shift
        mae = tf.reduce_mean(tf.abs(y - output))
        return mae

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data

        with tf.GradientTape() as tape:
            output = self(inputs, training=True)
            mse, block = self._compute_loss(y, output)
            loss = mse + block
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        mae = self._compute_metric(y, output)
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(mae)
        self.block_tracker.update_state(block)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result(),
            "block": self.block_tracker.result(),
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
        output = self(inputs, training=False)
        mse, block = self._compute_loss(y, output)
        loss = mse + block

        mae = self._compute_metric(y, output)
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(mae)
        self.block_tracker.update_state(block)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result(),
            "block": self.block_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def randomize_blocks(self, blocks):
        seq_dim = tf.shape(blocks)[0]

        block_mask = blocks > 0
        blocks = blocks[block_mask]
        num_blocks = tf.shape(blocks)[0]
        block_indices = tf.range(num_blocks, dtype=tf.int32)

        # 75 percent chance randomize sequence
        randomize_sequence = tf.cast(tf.random.uniform([1]) > 0.25, dtype=tf.int32)

        # mark upto 10 percent of the blocks as randomized
        random_mask = create_random_mask(
            [num_blocks],
            percent=tf.random.uniform([], minval=0, maxval=0.1),
            dtype=tf.int32,
        )

        # zero out and swap randomized bloxks
        output_indices = block_indices * (1 - random_mask * randomize_sequence)
        mask_indices = tf.cast(tf.where(random_mask), dtype=tf.int32)
        shuffled_inidces = tf.random.shuffle(tf.squeeze(mask_indices, axis=-1))
        shuffled_inidces = tf.scatter_nd(mask_indices, shuffled_inidces, [num_blocks])
        output_indices = output_indices + shuffled_inidces * randomize_sequence

        # move upto 10 percent of valid block positions to non block positions
        original_block_indices = tf.cast(tf.where(block_mask), dtype=tf.int32)
        non_block_indices = tf.cast(tf.where(True ^ block_mask), dtype=tf.int32)
        non_block_indices = tf.random.shuffle(non_block_indices)[:num_blocks]
        moved_masked = create_random_mask(
            [num_blocks, 1],
            percent=tf.random.uniform([], minval=0, maxval=0.1),
            dtype=tf.int32,
        )

        # zero out and replace block indices with non block indices
        new_block_indices = original_block_indices * (
            1 - moved_masked * randomize_sequence
        )
        new_block_indices = (
            new_block_indices + non_block_indices * moved_masked * randomize_sequence
        )

        # reconstruct dense output
        output_blocks = tf.gather(blocks, output_indices)
        output_blocks = tf.scatter_nd(new_block_indices, output_blocks, [seq_dim])

        # construct positions of altererd blocks
        positions = tf.scatter_nd(new_block_indices, random_mask, [seq_dim])

        # mark positions of blocks that were moved to non block positions
        moved_positions = tf.scatter_nd(
            original_block_indices, tf.squeeze(moved_masked, axis=-1), [seq_dim]
        )
        positions = positions | moved_positions

        return output_blocks, tf.cast(positions, dtype=tf.float32)

    def call(
        self, inputs, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        sparse_indices, sample_embeddings, dense_counts = inputs

        sample_embeddings = self.base_model(
            [sparse_indices, sample_embeddings], training=training
        )
        sample_embeddings = sample_embeddings / tf.norm(
            sample_embeddings, axis=-1, keepdims=True
        )
        dense_counts = tf.cast(dense_counts, dtype=tf.float32)
        dense_counts = tf.math.log1p(
            dense_counts / tf.reduce_sum(dense_counts, axis=-1, keepdims=True)
        )
        dense_counts = tf.cast(dense_counts, dtype=self.compute_dtype)
        shuffled_counts, positions = tf.map_fn(
            self.randomize_blocks,
            dense_counts,
            fn_output_signature=(
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([None], dtype=tf.float32),
            ),
        )
        if training:
            dense_counts = shuffled_counts
        else:
            positions = tf.zeros_like(positions)

        dense_counts = tf.expand_dims(dense_counts, axis=-1)
        positions = tf.expand_dims(positions, axis=-1)
        extractor_output, positions = self.count_extractor((dense_counts, positions))
        positions = tf.squeeze(positions, axis=-1)
        positions = tf.cast(positions > 0, dtype=tf.int32)
        sample_embeddings = tf.expand_dims(sample_embeddings, axis=1)

        encoder_input = sample_embeddings + extractor_output
        encoder_output = self.encoder(encoder_input)
        block_pred = self.block_pred(encoder_output)

        output = self.regressor(encoder_output)
        print("RegressorV2 exit...")
        return (output, positions, block_pred)

    def get_config(self):
        config = super(RegressorV2, self).get_config()
        config.update(
            {
                "shift": self.shift,
                "scale": self.scale,
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        print("Constructing RegressorV2 from config")
        input_shape = None
        base_model = tf.keras.saving.deserialize_keras_object(config["base_model"])
        base_model.trainable = False
        config["base_model"] = base_model
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)
        model.build(input_shape)
        return model
