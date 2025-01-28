from __future__ import annotations

from typing import Iterable, Union, List

import numpy as np
import tensorflow as tf
from biom import Table, load_table
from bp import parse_newick
from skbio import DistanceMatrix

from aam.data_handlers.unifrac_generator import UniFracGenerator


class GOTUGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        gotu_table: Union[str, Table],
        asv_table: Union[str, Table],
        tree_path=None,
        gen_new_table_frequency=3,
        batch_size=4,
        shuffle=False,
        return_sample_ids=False,
        epochs=1000,
        asv_rarefy_depth=1000,
        gotu_rarefy_depth=100000,
        **kwargs,
    ):
        kwargs["tree_path"] = tree_path
        kwargs["table"] = asv_table
        kwargs["rarefy_depth"] = asv_rarefy_depth
        kwargs["is_16S"] = True
        self.asv_generator = UniFracGenerator(**kwargs)
        kwargs["table"] = gotu_table
        kwargs["rarefy_depth"] = gotu_rarefy_depth
        kwargs["is_16S"] = False
        self.gotu_generator = UniFracGenerator(**kwargs)
        self.generators = [self.asv_generator, self.gotu_generator]
        self.common_ids = np.intersect1d(
            self.generators[0].sample_ids,
            self.generators[1].sample_ids,
            assume_unique=True,
        )
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
        def _get_data(gen):
            tokens, sparse_indices, counts = [], [], []
            y_true, encoder_output = [], []
            (
                (gen_tokens, gen_sparse_indices, gen_obs_indices, gen_counts),
                (gen_y_true, gen_encoder_output),
            ) = gen._batch_data(batch_sample_ids)
            gen_tokens = gen_tokens[gen_obs_indices]
            tokens.append(gen_tokens)

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
            return (
                encoder_output,
                unique_tokens,
                sparse_indices,
                obs_indices,
                counts,
                y_true,
            )

        encoder_output, unique_tokens, sparse_indices, obs_indices, counts, y_true = (
            _get_data(self.asv_generator)
        )
        (
            gotu_encoder_output,
            gotu_unique_tokens,
            gotu_sparse_indices,
            gotu_obs_indices,
            gotu_counts,
            gotu_y_true,
        ) = _get_data(self.gotu_generator)
        return (
            (
                (unique_tokens, sparse_indices, obs_indices, counts),
                (y_true, encoder_output),
            ),
            (
                (
                    gotu_unique_tokens,
                    gotu_sparse_indices,
                    gotu_obs_indices,
                    gotu_counts,
                ),
                (gotu_y_true, gotu_encoder_output),
            ),
        )


def get_dataset(gen: GOTUGenerator):
    enqueuer = tf.keras.utils.OrderedEnqueuer(gen, use_multiprocessing=True)
    enqueuer.start(workers=2, max_queue_size=2 * gen.steps_per_epoch)
    gen.stop = enqueuer.stop

    if not gen.return_sample_ids:
        y_type = tf.TensorSpec(
            shape=[gen.batch_size, gen.batch_size],
            dtype=tf.float32,
        )
    else:
        y_type = tf.TensorSpec(shape=(gen.batch_size), dtype=tf.string)

    dataset = tf.data.Dataset.from_generator(
        enqueuer.get,
        output_signature=(
            (
                (
                    tf.TensorSpec(shape=[None, 150], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
                    tf.TensorSpec(shape=[None], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
                ),
                (
                    tf.TensorSpec(
                        shape=[gen.batch_size, 1],
                        dtype=tf.float32,
                    ),
                    y_type,
                ),
            ),
            (
                (
                    tf.TensorSpec(shape=[None], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
                    tf.TensorSpec(shape=[None], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
                ),
                (
                    tf.TensorSpec(
                        shape=[gen.batch_size, 1],
                        dtype=tf.float32,
                    ),
                    y_type,
                ),
            ),
        ),
    )

    return dataset


if __name__ == "__main__":
    gotu_gen = GOTUGenerator(
        gotu_table="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/gotu_ordered_table.biom",
        asv_table="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/asv_ordered_table.biom",
        tree_path="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/tulsa-tree.nwk",
        metadata="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/metag_metadata.tsv",
        metadata_column="host_age",
        shift=0.0,
        scale=100.0,
        gen_new_tables=True,
        is_16S=False,
        asv_rarefy_depth=1000,
        gotu_rarefy_depth=100000,
    )
    dataset = get_dataset(gotu_gen)
    for x, y in dataset.take(1):
        print(x)
        print(y)
        break

    gotu_gen.stop()
