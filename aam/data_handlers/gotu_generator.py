from __future__ import annotations

import json
from typing import Iterable, List, Union

import numpy as np
import tensorflow as tf
from biom import Table, load_table
from bp import parse_newick
from skbio import DistanceMatrix

from aam.data_handlers.unifrac_generator import UniFracGenerator


def load_json(fp: str) -> dict:
    with open(fp) as f:
        output = json.load(f)
    return output


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
        gotu_tree_index=None,
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
        self.gotu_tree_index = load_json(gotu_tree_index)
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
        (
            (asv_unique_tokens, asv_sparse_indices, asv_obs_indices, asv_counts),
            (asv_y_true, asv_encoder_output),
        ) = self.asv_generator._batch_data(batch_sample_ids)
        (
            (gotu_unique_tokens, gotu_sparse_indices, gotu_obs_indices, gotu_counts),
            (gotu_y_true, gotu_encoder_output),
        ) = self.gotu_generator._batch_data(batch_sample_ids)

        gotu_ids = self.gotu_generator._rarefied_table.ids(axis="observation")
        gotu_node_ids = [gotu_ids[i] for i in gotu_unique_tokens]
        gotu_unique_tokens = (
            np.array([self.gotu_tree_index[id] for id in gotu_node_ids], dtype=np.int32)
            + 3
        )
        return (
            (
                asv_unique_tokens,
                asv_sparse_indices,
                asv_obs_indices,
                asv_counts,
                gotu_unique_tokens,
                gotu_sparse_indices,
                gotu_obs_indices,
                gotu_counts,
            ),
            (
                (asv_y_true, asv_encoder_output),
                (gotu_y_true, gotu_encoder_output),
            ),
        )


def get_dataset(gen: GOTUGenerator):
    enqueuer = tf.keras.utils.OrderedEnqueuer(gen, use_multiprocessing=True)
    enqueuer.start(workers=1, max_queue_size=2 * gen.steps_per_epoch)
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
                tf.TensorSpec(shape=[None, 150], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            ),
            (
                (
                    tf.TensorSpec(
                        shape=[gen.batch_size, 1],
                        dtype=tf.float32,
                    ),
                    y_type,
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
    print(gotu_gen[0])

    gotu_gen.stop()
