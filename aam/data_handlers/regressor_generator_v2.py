from __future__ import annotations

from functools import wraps
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table

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


class RegressorGeneratorV2(tf.keras.utils.Sequence):
    def __init__(
        self,
        table: Union[str, Table] = None,
        metadata: Optional[Union[str, pd.DataFrame]] = None,
        metadata_column: Optional[str] = None,
        sequence_embeddings: Optional[str] = None,
        sequence_labels: Optional[str] = None,
        sample_embeddings: Optional[str] = None,
        sample_labels: Optional[str] = None,
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
        shift=None,
        scale=None,
    ):
        self.sequence_embeddings = SequenceEmbeddings(
            sequence_embeddings, sequence_labels, normalize_embeddings
        )
        self.sample_embeddings = SequenceEmbeddings(
            sample_embeddings, sample_labels, normalize_embeddings
        )
        if isinstance(table, str):
            table = load_table(table)

        self.table: Table = table

        print("original table shape:", self.table.shape)
        self.asv_ids = self.table.ids(axis="observation")
        self.num_asvs = len(self.table.ids(axis="observation"))

        self.metadata_column: str = metadata_column
        self.metadata: pd.Series = metadata
        if shift is None:
            self.shift = np.mean(self.metadata[self.metadata_column])
            self.scale = np.std(self.metadata[self.metadata_column])
        else:
            self.shift = shift
            self.scale = scale
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
        return self._batch_data(self.sample_ids[start:end])

    def _batch_data(self, batch_sample_ids):
        sparse_indices, embeddings, dense_counts = [], [], []
        for batch_i, s_id in enumerate(batch_sample_ids):
            counts = self.rarefied_table.data(s_id, dense=True)

            obs_indices = np.squeeze(np.argwhere(counts > 0), axis=-1)
            sparse_indices.append([[batch_i, i] for i in range(len(obs_indices))])
            embeddings.append(self.sequence_embeddings[obs_indices])
            # embeddings.append(np.mean(self.sequence_embeddings[obs_indices], axis=0))
            dense_counts.append(counts)

        dense_counts = np.vstack(dense_counts)
        sparse_indices = np.vstack(sparse_indices)
        embeddings = np.vstack(embeddings)
        # embeddings = self.sample_embeddings.get(batch_sample_ids)
        y_true = (
            self.metadata.loc[batch_sample_ids, self.metadata_column]
            .to_numpy()
            .reshape((-1, 1))
        )

        if self.return_sample_ids:
            return (sparse_indices, embeddings, dense_counts), batch_sample_ids

        return (sparse_indices, embeddings, dense_counts), (
            y_true - self.shift
        ) / self.scale

    def on_epoch_end(self):
        if (
            self.gen_new_tables
            and self.epochs_since_last_table >= self.gen_new_table_frequency
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
        print("finishing processing rarefied table...")
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
    from aam.models.regressor_v2 import RegressorV2

    ug = RegressorGeneratorV2(
        table="/home/kalen/removing-study-id/healthy-us-sorted-table.biom",
        metadata="/home/kalen/removing-study-id/healthy-us-metadata-train.tsv",
        metadata_column="host_age_normalized_years",
        sequence_embeddings="/home/kalen/removing-study-id/healthy-us-sequence-embeddings.npy",
        sequence_labels="/home/kalen/removing-study-id/healthy-us-sequence-labels.npy",
        sample_embeddings="/home/kalen/removing-study-id/healthy-us-sample-embeddings.npy",
        sample_labels="/home/kalen/removing-study-id/healthy-us-sample-labels.npy",
        gen_new_tables=True,
        shuffle=True,
        batch_size=8,
        return_sample_ids=False,
        drop_remainder=False,
        rarefy_depth=10000,
    )
    x, y = ug[0]
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)

    # model = RegressorV2(None, 0, 1)
    # model.build(
    #     [
    #         [None, 2],
    #         [None, ug.sample_embeddings.embeddings.shape[-1]],
    #         [None, ug.num_asvs],
    #     ]
    # )
    # model.summary()
    # # print(model(x))
    # # # # print(y)
