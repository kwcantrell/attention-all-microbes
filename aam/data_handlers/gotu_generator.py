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
        gotu_max_tokens=128,
        gotu_tree_index=None,
        **kwargs,
    ):
        kwargs["tree_path"] = tree_path
        kwargs["table"] = asv_table
        kwargs["rarefy_depth"] = asv_rarefy_depth
        kwargs["is_16S"] = True
        self.asv_generator = UniFracGenerator(**kwargs)
        if isinstance(gotu_table, str):
            gotu_table = load_table(gotu_table)
        self.gotu_table = gotu_table
        self.common_ids = np.intersect1d(
            self.asv_generator.sample_ids,
            self.gotu_table.ids(axis="sample"),
            assume_unique=True,
        )
        self.gotu_tree_index = load_json(gotu_tree_index)
        self.gotu_tree_index.update({-1: 0})
        self.size = len(self.common_ids)
        self.sample_indices = np.arange(self.size)

        self.batch_size = batch_size
        self.steps_per_epoch = self.size // self.batch_size
        self.shuffle = shuffle
        self.gen_new_table_frequency = gen_new_table_frequency
        self.epochs_since_last_table = 0
        self.return_sample_ids = return_sample_ids
        self.epochs = epochs
        self.gotu_obs_ids = self.gotu_table.ids(axis="observation")
        self.gotu_max_tokens = gotu_max_tokens
        self.sample_indices = np.arange(self.size, dtype=np.int32)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.asv_generator.on_epoch_end()

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

    def _batch_gotu_data(self, batch_sample_ids):
        num_unique_asvs, sparse_indices, obs_indices, counts = [], [], [], []
        for s_id in batch_sample_ids:
            sample_data = self.gotu_table.data(s_id, dense=False).tocoo()
            (obs_idx, _), sample_counts = sample_data.coords, sample_data.data

            # remove zeros
            non_zero_mask = sample_counts > 0.0
            obs_idx = obs_idx[non_zero_mask]
            sample_counts = sample_counts[non_zero_mask]
            sorted_indices = np.argsort(sample_counts)
            sorted_indices = sorted_indices[::-1]
            obs_idx = obs_idx[sorted_indices]
            sample_counts = sample_counts[sorted_indices]
            if len(sample_counts) < self.gotu_max_tokens:
                pad_amount = self.gotu_max_tokens - len(sample_counts)
                sample_counts = np.pad(sample_counts, pad_width=(0, pad_amount))
                obs_idx = np.pad(obs_idx, pad_width=(0, pad_amount), constant_values=-1)
            obs_indices.append(obs_idx[: self.gotu_max_tokens])
            counts.append(sample_counts[: self.gotu_max_tokens])

        obs_indices = np.vstack(obs_indices, dtype=np.int32)
        sample_counts = np.vstack(counts, dtype=np.float32)

        obs_ids = self.gotu_obs_ids[obs_indices]

        def _cast_to_tokens(sample_obs_ids):
            tokens = [self.gotu_tree_index[obs_id] for obs_id in sample_obs_ids]
            tokens = np.array(tokens, dtype=np.int32)
            return tokens

        tokens = [_cast_to_tokens(sample_obs_ids) for sample_obs_ids in obs_ids]
        tokens = np.vstack(tokens, dtype=np.int32)

        return tokens[: self.gotu_max_tokens], sample_counts[: self.gotu_max_tokens]

    def _batch_data(self, batch_sample_ids):
        (
            (asv_unique_tokens, asv_sparse_indices, asv_obs_indices, asv_counts),
            (asv_y_true, asv_encoder_output),
        ) = self.asv_generator._batch_data(batch_sample_ids)
        gotu_tokens, gotu_counts = self._batch_gotu_data(batch_sample_ids)
        return (
            asv_unique_tokens,
            asv_sparse_indices,
            asv_obs_indices,
            asv_counts,
            gotu_tokens,
            gotu_counts,
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
            tf.TensorSpec(shape=[None, 150], dtype=tf.int32),
            tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            tf.TensorSpec(shape=[None, gen.gotu_max_tokens], dtype=tf.int32),
            tf.TensorSpec(shape=[None, gen.gotu_max_tokens], dtype=tf.float32),
        ),
    )

    return dataset


if __name__ == "__main__":
    gotu_gen = GOTUGenerator(
        gotu_table="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/gotu_ordered_table.biom",
        asv_table="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/asv_ordered_table.biom",
        tree_path="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/tulsa-tree.nwk",
        metadata="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/metag_metadata.tsv",
        gotu_tree_index="/home/jokirkland/data/trees/gotu_node_dict.json",
        metadata_column="host_age",
        shift=0.0,
        scale=100.0,
        gen_new_tables=True,
        is_16S=False,
        asv_rarefy_depth=1000,
        gotu_max_tokens=128,
    )
    dataset = get_dataset(gotu_gen)
    print(gotu_gen[0])

    gotu_gen.stop()
