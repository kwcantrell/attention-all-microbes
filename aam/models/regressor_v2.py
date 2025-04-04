from __future__ import annotations

import math
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
        pooling_size=2048,
        **kwargs,
    ):
        super(RegressorV2, self).__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_tracker = tf.keras.metrics.Mean(name="mae")
        self.counts_tracker = tf.keras.metrics.Mean(name="pair")
        self.block_tracker = tf.keras.metrics.Mean(name="pair")
        self.pair_loss = PairwiseLoss()
        self.block_entropy = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE, label_smoothing=0.1
        )

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

        sparse_indices, sample_embeddings, dense_counts = input_shape
        self.num_blocks = math.ceil(dense_counts[-1] / self.pooling_size)
        self.count_extractor = tf.keras.Sequential(
            [
                PoolingBlock(self.num_filters, self.pooling_size),
            ],
            name="count_extractor",
        )
        self.block_pred = tf.keras.Sequential(
            [
                FeedForward(),
                tf.keras.layers.Dense(
                    self.num_blocks, dtype=tf.float32, activation="softmax"
                ),
            ],
            name="block_pred",
        )

        self.reconstruct_counts = tf.keras.Sequential(
            [
                FeedForward(),
                tf.keras.layers.Dense(units=self.num_filters, dtype=tf.float32),
            ],
            name="reconstruct_counts",
        )
        self.encoder = TransformerEncoder(
            num_layers=8,
            num_attention_heads=8,
            intermediate_size=1024,
            use_linear_bias=True,
            name="encoder",
        )
        self.regressor = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
                FeedForward(),
                tf.keras.layers.Dense(units=1, dtype=tf.float32),
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
        output = self(inputs, training=False)
        y = y * self.scale + self.shift
        output = output * self.scale + self.shift
        return output, y

    def _compute_loss(self, y, output):
        mse = tf.reduce_mean(tf.abs(y - output))
        counts_loss, position_loss = self.losses
        return mse, counts_loss, position_loss

    def _compute_metric(self, y, output):
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
            output = self(inputs, compute_transormer_loss=True, training=True)
            mse, counts, block = self._compute_loss(y, output)
            unscaled_loss = mse + counts + block

            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(unscaled_loss)
            else:
                loss = unscaled_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        mae = self._compute_metric(y, output)
        self.loss_tracker.update_state(unscaled_loss)
        self.mae_tracker.update_state(tf.reduce_mean(mae))
        self.block_tracker.update_state(tf.reduce_mean(block))
        self.counts_tracker.update_state(tf.reduce_mean(counts))
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result(),
            "block": self.block_tracker.result(),
            "counts": self.counts_tracker.result(),
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
        output = self(inputs, compute_transormer_loss=True, training=False)
        mse, counts, block = self._compute_loss(y, output)
        loss = mse + counts + block

        mae = self._compute_metric(y, output)
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(tf.reduce_mean(mae))
        self.block_tracker.update_state(tf.reduce_mean(block))
        self.counts_tracker.update_state(tf.reduce_mean(counts))
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result(),
            "block": self.block_tracker.result(),
            "counts": self.counts_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def randomize_blocks(self, original_blocks, training):
        original_blocks = tf.cast(original_blocks, dtype=tf.float32)
        original_blocks = tf.cast(
            original_blocks / tf.reduce_sum(original_blocks, axis=-1, keepdims=True),
            dtype=self.compute_dtype,
        )
        original_blocks = tf.cast(original_blocks, dtype=self.compute_dtype)
        seq_dim = tf.shape(original_blocks)[0]
        block_mask = original_blocks > 0
        blocks = original_blocks[block_mask]
        num_blocks = tf.shape(blocks)[0]
        block_indices = tf.range(num_blocks, dtype=tf.int32)

        if not training:
            block_indices = tf.cast(tf.where(block_mask), dtype=tf.int32)
            return (
                tf.scatter_nd(block_indices, blocks, [seq_dim]),
                tf.zeros([seq_dim], dtype=self.compute_dtype),
            )

        # 90 percent chance randomize sequence
        randomize_sequence = tf.cast(tf.random.uniform([1]) > 0.01, dtype=tf.int32)

        # mark upto 5 percent of the blocks as randomized
        random_mask = create_random_mask([num_blocks], percent=0.03, dtype=tf.int32)

        # zero out and swap randomized bloxks
        output_indices = block_indices * (1 - random_mask * randomize_sequence)
        mask_indices = tf.cast(tf.where(random_mask), dtype=tf.int32)
        shuffled_inidces = tf.random.shuffle(tf.squeeze(mask_indices, axis=-1))
        shuffled_inidces = tf.scatter_nd(mask_indices, shuffled_inidces, [num_blocks])
        output_indices = output_indices + shuffled_inidces * randomize_sequence

        # move upto 5 percent of valid block positions to non block positions
        original_block_indices = tf.cast(tf.where(block_mask), dtype=tf.int32)
        non_block_indices = tf.cast(tf.where(~block_mask), dtype=tf.int32)
        non_block_indices = tf.random.shuffle(non_block_indices)[:num_blocks]
        moved_masked = create_random_mask([num_blocks, 1], percent=0.01, dtype=tf.int32)

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
        positions = original_blocks != output_blocks
        return output_blocks, tf.cast(positions, dtype=self.compute_dtype)

    def call(
        self, inputs, compute_transormer_loss=False, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        sparse_indices, sample_embeddings, dense_counts = inputs

        sample_embeddings = self.base_model(
            [sparse_indices, sample_embeddings], training=False
        )
        # sample_embeddings = sample_embeddings / tf.norm(
        #     sample_embeddings, axis=-1, keepdims=True
        # )
        sample_embeddings = sample_embeddings / self.num_blocks

        modified_counts, positions = tf.map_fn(
            lambda x: self.randomize_blocks(x, training=training),
            dense_counts,
            fn_output_signature=(
                tf.TensorSpec([None], dtype=self.compute_dtype),
                tf.TensorSpec([None], dtype=self.compute_dtype),
            ),
        )

        sample_embeddings = tf.expand_dims(sample_embeddings, axis=1)
        dense_counts = tf.expand_dims(dense_counts, axis=-1)
        modified_counts = tf.expand_dims(modified_counts, axis=-1)
        positions = tf.expand_dims(positions, axis=-1)
        pooled_counts, extractor_output, positions = self.count_extractor(
            (dense_counts, modified_counts, sample_embeddings, positions)
        )

        # predict true block
        pooled_counts = tf.cast(pooled_counts, dtype=tf.float32)
        block_pred = self.block_pred(pooled_counts)
        pooled_counts = tf.stop_gradient(pooled_counts)

        encoder_output = self.encoder(extractor_output)
        counts_pred = self.reconstruct_counts(encoder_output)
        output = self.regressor(encoder_output)

        # in addition to marked blocks, set an additional 10 percent as observers
        observe_mask = create_random_mask(
            tf.shape(positions),
            percent=0.25,
            dtype=tf.bool,
        )
        observe_mask = observe_mask | (positions > 0)
        positions = tf.cast(positions, dtype=tf.int32)

        if compute_transormer_loss:
            counts_loss = tf.reduce_mean(
                tf.reduce_mean(tf.square(pooled_counts - counts_pred), axis=1),
                axis=-1,
            )
            self.add_loss(tf.reduce_mean(counts_loss, axis=-1))

            block_shape = tf.shape(block_pred)
            batch_dim = block_shape[0]
            num_blocks = block_shape[-1]
            positions = tf.repeat(
                tf.expand_dims(tf.range(num_blocks), axis=0),
                repeats=batch_dim,
                axis=0,
            )
            positions = tf.one_hot(positions, depth=num_blocks)
            position_loss = self.block_entropy(positions, block_pred)
            position_loss = tf.reduce_mean(position_loss, axis=-1)
            self.add_loss(tf.reduce_mean(position_loss))

        print("RegressorV2 exit...")
        return output

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
