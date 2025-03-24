from __future__ import annotations

from functools import wraps
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table
from bp import parse_newick, to_skbio_treenode
from sklearn import preprocessing

from aam.data_handlers.sequence_embeddings import SequenceEmbeddings


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


class TripletGeneratorV2(tf.keras.utils.Sequence):
    def __init__(
        self,
        table: Union[str, Table] = None,
        metadata: Optional[Union[str, pd.DataFrame]] = None,
        metadata_column: Optional[str] = None,
        group_counts=None,
        sequence_embeddings: Optional[str] = None,
        sequence_labels: Optional[str] = None,
        normalize_embeddings: bool = False,
        shuffle: bool = False,
        rarefy_depth: int = 1000,
        epochs: int = 1000,
        gen_new_tables: bool = False,
        gen_new_table_frequency=3,
        return_sample_ids=False,
        seed=None,
        drop_remainder=True,
        batch_size=128,
    ):
        self.sequence_embeddings = SequenceEmbeddings(
            sequence_embeddings, sequence_labels, normalize_embeddings
        )
        if isinstance(table, str):
            table = load_table(table)

        self.table: Table = table

        print("original table shape:", self.table.shape)
        self.asv_ids = self.table.ids(axis="observation")
        self.num_asvs = len(self.table.ids(axis="observation"))

        self.metadata_column: str = metadata_column
        self.group_counts = group_counts
        self.metadata: pd.Series = metadata
        self.rarefy_depth: int = rarefy_depth
        self.return_sample_ids: bool = return_sample_ids

        self.shuffle = shuffle
        self.epochs = epochs
        self.gen_new_tables = gen_new_tables

        self.seed = seed
        self.gen_new_table_frequency = gen_new_table_frequency
        self.epochs_since_last_table = 0

        self.encoder_target = None
        self.encoder_dtype = None
        self.encoder_output_type = None
        self.sample_ids = None

        print("rarefy table...")
        self.rarefied_table: Table = self.table.subsample(rarefy_depth, seed=42)
        self.size = self.rarefied_table.shape[1]
        self.on_epoch_end()

        self.drop_remainder = drop_remainder
        self.batch_size = batch_size
        self.steps_per_epoch = self.size // self.batch_size
        if (
            not self.drop_remainder
            and self.steps_per_epoch * self.batch_size < self.size
        ):
            self.steps_per_epoch += 1

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        return self._batch_data(self.sample_ids[start:end], 1.0)

    def _batch_data(self, batch_sample_ids, weights):
        (
            num_unique_asvs,
            sparse_indices,
            obs_indices,
            counts,
            asv_counts,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )
        cur_row_indx = 0
        for s_id in batch_sample_ids:
            sample_data = self.rarefied_table.data(s_id, dense=False).tocoo()
            (obs_idx, _), sample_asv_counts = sample_data.coords, sample_data.data

            # remove zeros
            non_zero_mask = sample_asv_counts > 0.0
            obs_idx = obs_idx[non_zero_mask]
            sample_counts = sample_asv_counts[non_zero_mask]
            num_unique_asvs.append(len(obs_idx))
            sparse_indices.append([[cur_row_indx, i] for i in range(len(obs_idx))])

            obs_indices.append(obs_idx)
            counts.append(sample_counts)

            dense_counts = np.zeros(self.num_asvs)
            dense_counts[obs_idx] = sample_counts
            asv_counts.append(dense_counts)
            cur_row_indx += 1

        num_unique_asvs = np.array(num_unique_asvs, dtype=np.int32)
        sparse_indices = np.vstack(sparse_indices, dtype=np.int32)
        obs_indices = np.hstack(obs_indices, dtype=np.int32)
        counts = np.hstack(counts, dtype=np.float32)[:, np.newaxis]
        asv_counts = np.vstack(asv_counts)

        # get list of unique observations in batch
        unique_obs, obs_indices = np.unique(obs_indices, return_inverse=True)
        tokens = self.sequence_embeddings[unique_obs]
        y_true = self.metadata.loc[batch_sample_ids, self.metadata_column].to_numpy()

        if self.return_sample_ids:
            return (
                tokens,
                sparse_indices,
                obs_indices,
                counts,
                asv_counts,
            ), batch_sample_ids

        return (
            tokens,
            sparse_indices,
            obs_indices,
            counts,
            asv_counts,
        ), (
            y_true[:, np.newaxis],
            np.array([1 / self.group_counts[g] for g in y_true]),
        )

    def on_epoch_end(self):
        if (
            self.gen_new_tables
            and self.epochs_since_last_table > self.gen_new_table_frequency
        ):
            print("resampling dataset...")
            self.rarefied_table = self.table.subsample(self.rarefy_depth)
            self.epochs_since_last_table = 0

        if self.shuffle:
            np.random.shuffle(self.sample_ids)

        self.epochs_since_last_table += 1

    @property
    def rarefied_table(self):
        return self._rarefied_table

    @rarefied_table.setter
    def rarefied_table(self, rarefied_table: Table):
        self._metadata = self._metadata.loc[rarefied_table.ids()]
        self.sample_ids = rarefied_table.ids()
        self.sample_indices = np.arange(len(self.sample_ids))

        self._rarefied_table = self.sequence_embeddings.align_table(rarefied_table)

        print("creating encoder target...")

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
        metadata = metadata.loc[self.table.ids()]
        print(f"aligned table shape: {self.table.shape}")
        print(f"aligned metadata shape: {metadata.shape}")
        self._metadata = metadata.reindex(self.table.ids())
        print("done preprocessing metadata")


if __name__ == "__main__":
    df = pd.read_csv(
        "/home/kalen/removing-study-id/healthy-us-metadata-train.tsv",
        sep="\t",
        index_col=0,
    )
    group_counts = df["numeric_sequence_group"].value_counts().to_dict()
    ug = TripletGeneratorV2(
        table="/home/kalen/removing-study-id/healthy-us-sorted-table.biom",
        metadata="/home/kalen/removing-study-id/healthy-us-metadata-train.tsv",
        metadata_column="numeric_sequence_group",
        group_counts=group_counts,
        sequence_embeddings="/home/kalen/removing-study-id/healthy-us-sequence-embeddings.npy",
        sequence_labels="/home/kalen/removing-study-id/healthy-us-sequence-labels.npy",
        gen_new_tables=True,
        shuffle=True,
        batch_size=8,
        return_sample_ids=False,
        drop_remainder=False,
    )
    x, y = ug[0]
    print(x[0])
    print(y)
