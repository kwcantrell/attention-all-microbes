from __future__ import annotations

import math
import os
from functools import wraps
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table
from bp import parse_newick, to_skbio_treenode
from sklearn import preprocessing


def add_lock(func):
    lock = f"_{func.__name__}_lock"

    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        if not hasattr(obj, lock):
            setattr(obj, lock, True)
            return func(obj, *args, **kwargs)

        if getattr(obj, lock):
            raise Exception(f"Attempting to modify locked property '{func.__name__}'")

        setattr(obj, lock, True)
        return func(obj, *args, **kwargs)

    return wrapper


def batch_embeddings(asv_embeddings, batch_indicies, counts, asv_indices=None):
    emb_dim = tf.shape(asv_embeddings)[-1]
    if asv_indices is not None:
        asv_embeddings = tf.gather(asv_embeddings, asv_indices)
    batch_shape = tf.reduce_max(batch_indicies[:, 0]) + 1
    max_unique = tf.reduce_max(batch_indicies[:, 1]) + 1
    batch_embeddings = tf.scatter_nd(
        batch_indicies, asv_embeddings, shape=[batch_shape, max_unique, emb_dim]
    )
    counts = tf.scatter_nd(batch_indicies, counts, shape=[batch_shape, max_unique, 1])
    return batch_embeddings, counts


class GeneratorDatasetV2(tf.keras.utils.Sequence):
    taxon_field = "Taxon"
    levels = [f"Level {i}" for i in range(1, 8)]

    def __init__(
        self,
        table: Union[str, Table] = None,
        taxonomy: Optional[Union[str, pd.DataFrame]] = None,
        metadata: Optional[Union[str, pd.DataFrame]] = None,
        metadata_column: Optional[str] = None,
        shift: Optional[Union[str, float]] = None,
        scale: Union[str, float] = "minmax",
        max_token_per_sample: int = 1024,
        shuffle: bool = False,
        rarefy_depth: int = 5000,
        epochs: int = 1000,
        gen_new_tables: bool = False,
        batch_size: int = 8,
        max_bp: int = 150,
        is_16S: bool = True,
        is_categorical: Optional[bool] = None,
        gen_new_table_frequency=3,
        return_sample_ids=False,
        tree_path=None,
        seed=None,
    ):
        if isinstance(table, str):
            table = load_table(table)

        self.table: Table = table
        self.tree_path = tree_path
        self.is_categorical: bool = is_categorical
        self.metadata_column: str = metadata_column
        self.shift = shift
        self.scale = scale
        self.metadata: pd.Series = metadata
        self.rarefy_depth: int = rarefy_depth
        self.max_token_per_sample: int = max_token_per_sample
        self.return_sample_ids: bool = return_sample_ids

        self.include_sample_weight: bool = is_categorical

        self.shuffle = shuffle
        self.epochs = epochs
        self.gen_new_tables = gen_new_tables
        self.samples_per_minibatch = batch_size

        self.batch_size = batch_size
        self.max_bp = max_bp
        self.is_16S = is_16S
        self.seed = seed
        self.gen_new_table_frequency = gen_new_table_frequency
        self.epochs_since_last_table = 0

        self.encoder_target = None
        self.encoder_dtype = None
        self.encoder_output_type = None
        self.sample_ids = None
        self.asv_ids = None

        if self.tree_path is not None:
            self.tree = to_skbio_treenode(parse_newick(open(self.tree_path).read()))
            self.postorder_pos = {
                n.name: i for i, n in enumerate(self.tree.postorder()) if n.is_tip()
            }

        self.tax_level = f"Level {1}"
        self.taxonomy = taxonomy
        if taxonomy is not None:
            print("taxonomy info", self.num_tax_values)

        print("rarefy table...")
        self.rarefied_table: Table = self.table.subsample(rarefy_depth)

        self.size = self.rarefied_table.shape[1]
        self.steps_per_epoch = self.size // self.batch_size

        self.y_data = self.metadata.loc[self._rarefied_table.ids()]
        self.on_epoch_end()

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        sample_indices = self.sample_indices[start:end]
        batch_sample_ids = self.sample_ids[sample_indices]
        return self._batch_data(batch_sample_ids)

    def _batch_data(self, batch_sample_ids):
        num_unique_asvs, sparse_indices, obs_indices, counts, taxon_counts = (
            [],
            [],
            [],
            [],
            [],
        )
        cur_row_indx = 0
        for s_id in batch_sample_ids:
            sample_data = self.rarefied_table.data(s_id, dense=False).tocoo()
            (obs_idx, _), sample_counts = sample_data.coords, sample_data.data

            # remove zeros
            non_zero_mask = sample_counts > 0.0
            obs_idx = obs_idx[non_zero_mask]
            sample_counts = sample_counts[non_zero_mask]

            num_unique_asvs.append(len(obs_idx))
            sparse_indices.append([[cur_row_indx, i] for i in range(len(obs_idx))])

            obs_indices.append(obs_idx)
            counts.append(sample_counts)
            cur_row_indx += 1

            if self.taxonomy is not None:
                taxons = self.taxonomy.loc[self.asv_ids[obs_idx], "Taxon"].to_numpy()
                sample_tax_counts = np.zeros(self.num_tax_values)
                np.add.at(sample_tax_counts, taxons, sample_counts)
                taxon_counts.append(sample_tax_counts)

        num_unique_asvs = np.array(num_unique_asvs, dtype=np.int32)
        sparse_indices = np.vstack(sparse_indices, dtype=np.int32)
        obs_indices = np.hstack(obs_indices, dtype=np.int32)
        counts = np.hstack(counts, dtype=np.float32)[:, np.newaxis]

        if self.taxonomy is not None:
            taxon_counts = np.vstack(taxon_counts, dtype=np.float32)

        # get list of unique observations in batch
        unique_obs, obs_indices = np.unique(obs_indices, return_inverse=True)
        if self.is_16S:
            lookup = {
                "a": 1,
                "c": 2,
                "g": 3,
                "t": 4,
            }

            def map(asv):
                asv = asv.lower()
                return np.array([lookup[c] for c in asv], dtype=np.int32)[np.newaxis, :]

            tokens = np.concatenate(
                [map(asv) for asv in self.asv_ids[unique_obs]], axis=0
            )
        else:
            tokens = unique_obs
        y_true = self.y_data.loc[batch_sample_ids].to_numpy()[:, np.newaxis]

        if self.return_sample_ids:
            y = batch_sample_ids
        else:
            y = y_true

        if self.encoder_target is None:
            if self.taxonomy is None:
                return (tokens, sparse_indices, obs_indices, counts), y
            else:
                return (tokens, sparse_indices, obs_indices, counts, taxon_counts), y

        encoder_output = self._encoder_output(batch_sample_ids)
        return (tokens, sparse_indices, obs_indices, counts), (y_true, encoder_output)

    def on_epoch_end(self):
        if (
            self.gen_new_tables
            and self.epochs_since_last_table > self.gen_new_table_frequency
        ):
            print("resampling dataset...")
            self.rarefied_table = self.table.subsample(self.rarefy_depth)
            self.epochs_since_last_table = 0

        if self.shuffle:
            np.random.shuffle(self.sample_indices)

        self.epochs_since_last_table += 1

    @property
    def taxonomy(self):
        return self._taxonomy

    @taxonomy.setter
    def taxonomy(self, taxonomy):
        if taxonomy is None:
            self._taxonomy = taxonomy
            return

        taxonomy[self.levels] = taxonomy[self.taxon_field].str.split("; ", expand=True)
        taxonomy = taxonomy.loc[taxonomy[self.tax_level].str.len() > 3]
        taxonomy = taxonomy.loc[:, self.levels]
        taxonomy.loc[:, "Taxon"] = taxonomy.loc[:, self.levels[:6]].agg(
            "; ".join, axis=1
        )
        self.table = self.table.filter(
            set(self.table.ids(axis="observation")).intersection(set(taxonomy.index)),
            axis="observation",
        )
        self.table.remove_empty()

        print(taxonomy["Taxon"].to_numpy())

        le = preprocessing.LabelEncoder()
        taxonomy["Taxon"] = le.fit_transform(taxonomy.loc[taxonomy.index, "Taxon"])
        self.taxonomy_values = taxonomy["Taxon"].unique()
        self.num_tax_values = np.max(self.taxonomy_values) + 1
        self._taxonomy = taxonomy

    @property
    def rarefied_table(self):
        return self._rarefied_table

    @rarefied_table.setter
    def rarefied_table(self, table: Table):
        self._rarefied_table = table
        print("removing empty sample/obs from table")
        self._rarefied_table.remove_empty()
        if self.tree_path is not None:

            def sort_obs(obs):
                post_pos = [self.postorder_pos[ob] for ob in obs]
                sorted_indices = np.argsort(post_pos)
                return obs[sorted_indices]

            self._rarefied_table = self._rarefied_table.sort(
                sort_obs, axis="observation"
            )

        self.sample_ids = self._rarefied_table.ids()
        self.asv_ids = self._rarefied_table.ids(axis="observation")
        self.sample_indices = np.arange(len(self.sample_ids))

        print("creating encoder target...")
        self.encoder_target = self._create_encoder_target()
        print("encoder target created")

    def _create_encoder_target(self) -> None:
        return None

    def _encoder_output(self, sample_ids):
        return None

    @property
    def table(self) -> Table:
        return self._table

    @table.setter
    def table(self, table: Union[str, Table]):
        self._table = table

    @property
    def metadata(self) -> pd.Series:
        return self._metadata

    @metadata.setter
    @add_lock
    def metadata(self, metadata: Union[str, pd.DataFrame]):
        if metadata is None:
            return

        if isinstance(metadata, str):
            metadata = pd.read_csv(metadata, sep="\t", index_col=0, dtype={0: str})

        if self.metadata_column not in metadata.columns:
            raise Exception(f"Invalid metadata column {self.metadata_column}")

        print("aligning table with metadata")
        samp_ids = np.intersect1d(self.table.ids(axis="sample"), metadata.index)
        self.table.filter(samp_ids, axis="sample", inplace=True)
        self.table.remove_empty()
        metadata = metadata.loc[self.table.ids(), self.metadata_column]
        print(f"aligned table shape: {self.table.shape}")
        print(f"aligned metadata shape: {metadata.shape}")
        if not self.is_categorical:
            metadata = metadata.astype(np.float32)
            # if not isinstance(self.scale, (str, float)):
            #     raise Exception("Invalid scale argument.")
            # if self.shift is None and isinstance(self.scale, float):
            #     raise Exception("Invalid shift argument")

            if self.scale == "minmax":
                self.shift = np.min(metadata)
                self.scale = np.max(metadata) - self.shift
            elif self.scale == "standscale":
                self.shift = np.mean(metadata)
                self.scale = np.std(metadata)

            metadata = (metadata - self.shift) / self.scale
        self._metadata = metadata.reindex(self.table.ids())
        print("done preprocessing metadata")


def get_dataset(gen: GeneratorDatasetV2):
    enqueuer = tf.keras.utils.OrderedEnqueuer(gen, use_multiprocessing=True)
    enqueuer.start(workers=2, max_queue_size=gen.steps_per_epoch)
    gen.stop = lambda: enqueuer.stop(0.1)

    batch_dim = gen.samples_per_minibatch
    if not gen.return_sample_ids:
        y_type = tf.TensorSpec(shape=(batch_dim, 1), dtype=tf.float32)
    else:
        y_type = tf.TensorSpec(shape=(batch_dim), dtype=tf.string)

    dataset = tf.data.Dataset.from_generator(
        enqueuer.get,
        output_signature=(
            (
                tf.TensorSpec(shape=[None, 150], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
                tf.TensorSpec(shape=[gen.batch_size, gen.num_tax_values]),
            ),
            y_type,
        ),
    )
    return dataset


if __name__ == "__main__":
    import pandas as pd

    taxonomy = pd.read_csv(
        "/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-taxonomy.tsv",
        sep="\t",
        index_col=0,
    )
    ug = GeneratorDatasetV2(
        table="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom",
        metadata="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-healthy.txt",
        taxonomy=taxonomy,
        metadata_column="host_age",
        scale="minmax",
        gen_new_tables=True,
        max_token_per_sample=100,
        batch_size=4,
        rarefy_depth=1000,
        return_sample_ids=True,
    )
    dataset = get_dataset(ug)
    # x, y = ug[0]
    # print(x)
    # (tokens, batch_indices, obs_indices, counts) = x
    # print("tokens:", tokens.shape)
    # print("batch_indices:", batch_indices.shape, batch_indices)
    # print("obs indices:", obs_indices.shape)
    # print("counts:", counts)
    # print("y_true", y[0].shape)
    # print("encoder output", y[1].shape)
