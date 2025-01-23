from __future__ import annotations

from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table

from aam.data_handlers.generator_dataset import GeneratorDataset
from aam.data_handlers.unifrac_generator import UniFracGenerator


class MultiDepthGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        table: Union[str, Table],
        sample_depths: List[int],
        tree_path=None,
        unifrac_metric="unifrac",
        gen_new_table_frequency=3,
        batch_size=4,
        shuffle=False,
        return_sample_ids=False,
        epochs=1000,
        **kwargs,
    ):
        if isinstance(table, str):
            table = load_table(table)
        self.unifrac_metric = unifrac_metric
        kwargs["tree_path"] = tree_path
        if unifrac_metric is not None:
            kwargs["unifrac_metric"] = unifrac_metric
            self.generators = [
                UniFracGenerator(table=table, rarefy_depth=depth, shuffle=False, batch_size=batch_size, **kwargs)
                for depth in sample_depths
            ]
        else:
            self.generators = [
                GeneratorDataset(table=table, rarefy_depth=depth, shuffle=False, batch_size=batch_size, **kwargs)
                for depth in sample_depths
            ]
        self.common_ids = np.intersect1d(self.generators[0].sample_ids, self.generators[1].sample_ids, assume_unique=True)
        self.size = len(self.common_ids)
        self.sample_indices = np.arange(self.size)

        self.batch_size = batch_size
        self.steps_per_epoch = self.size // self.batch_size
        self.shuffle = shuffle
        self.gen_new_table_frequency = gen_new_table_frequency
        self.epochs_since_last_table = 0
        self.return_sample_ids = return_sample_ids
        self.epochs = epochs
        self.sample_indices = np.arange(self.size, dtype=np.int32)
        self.on_epoch_end()

    def on_epoch_end(self):
        for g in self.generators:
            g.on_epoch_end()

        if self.shuffle:
            np.random.shuffle(self.sample_indices)

        self.epochs_since_last_table += 1

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        sample_indices = self.sample_indices[start:end]
        batch_sample_ids = self.common_ids[sample_indices]
        return self._batch_data(batch_sample_ids)

    def _batch_data(self, batch_sample_ids):
        tokens, sparse_indices, counts = [], [], []
        y_true, encoder_output = [], []
        for gen_i, generator in enumerate(self.generators):
            (gen_tokens, gen_sparse_indices, gen_obs_indices, gen_counts), (gen_y_true, gen_encoder_output) = (
                generator._batch_data(batch_sample_ids)
            )
            gen_tokens = gen_tokens[gen_obs_indices]
            tokens.append(gen_tokens)

            gen_sparse_indices[:, 0] += len(batch_sample_ids) * gen_i
            sparse_indices.append(gen_sparse_indices)

            counts.append(gen_counts)

            y_true.append(gen_y_true)
            encoder_output.append(gen_encoder_output)

        tokens = np.concatenate(tokens, axis=0)
        sparse_indices = np.concatenate(sparse_indices, axis=0)
        counts = np.concatenate(counts, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        encoder_output = np.concatenate(encoder_output, axis=0)

        unique_tokens, obs_indices = np.unique(tokens, axis=0, return_inverse=True)

        if self.unifrac_metric:
            return (unique_tokens, sparse_indices, obs_indices, counts), (y_true, encoder_output)
        else:
            return (unique_tokens, sparse_indices, obs_indices, counts), y_true


def get_dataset(gen: MultiDepthGenerator):
    enqueuer = tf.keras.utils.OrderedEnqueuer(gen, use_multiprocessing=True)
    enqueuer.start(workers=2, max_queue_size=2 * gen.steps_per_epoch)
    gen.stop = enqueuer.stop

    if not gen.return_sample_ids:
        y_type = tf.TensorSpec(shape=[gen.batch_size * len(gen.generators), gen.batch_size], dtype=tf.float32)
    else:
        y_type = tf.TensorSpec(shape=(gen.batch_size * len(gen.generators)), dtype=tf.string)

    if gen.unifrac_metric:
        dataset = tf.data.Dataset.from_generator(
            enqueuer.get,
            output_signature=(
                (
                    tf.TensorSpec(shape=[None, 150], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
                    tf.TensorSpec(shape=[None], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
                ),
                (tf.TensorSpec(shape=[gen.batch_size * len(gen.generators), 1], dtype=tf.float32), y_type),
            ),
        )
    else:
        dataset = tf.data.Dataset.from_generator(
            enqueuer.get,
            output_signature=(
                (
                    tf.TensorSpec(shape=[None, 150], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
                    tf.TensorSpec(shape=[None], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
                ),
                tf.TensorSpec(shape=[gen.batch_size * len(gen.generators), 1], dtype=tf.float32),
            ),
        )

    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    import numpy as np

    ug = MultiDepthGenerator(
        table="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom",
        tree_path="/home/kalen/aam-research-exam/research-exam/agp/data/agp-aligned.nwk",
        metadata="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-healthy.txt",
        metadata_column="host_age",
        sample_depths=[100, 1000],
        # shift=0.0,
        scale="minmax",
        gen_new_tables=True,
        max_token_per_sample=2048,
        batch_size=4,
    )

    # dataset = get_dataset(ug)
    # for x in dataset.take(1):
    #     print(x)
    x, y = ug[0]
    (tokens, batch_indices, obs_indices, counts) = x
    print("tokens:", tokens.shape)
    print("batch_indices:", batch_indices.shape, batch_indices)
    print("obs indices:", obs_indices.shape)
    print("counts:", counts.shape)
