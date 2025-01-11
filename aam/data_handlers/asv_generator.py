from __future__ import annotations

import os
from functools import wraps
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table


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


def _matching_sample_indices(query, search):
    indices = np.arange(len(search), dtype=np.int32)
    search = np.expand_dims(search, axis=0)
    query = np.expand_dims(query, axis=1)
    mask = np.equal(query, search)
    mask = np.any(mask, axis=0)
    return mask, indices[mask]


# Unicode mapping dictionary
mapping = {65: 1, 67: 2, 71: 3, 84: 4}  # Maps Unicode numbers to specific values


# Create the mapping function
def map_unicode(val):
    return mapping.get(val, 0)  # Return 0 if the value is not in the mapping


class ASVGenerator:
    table_fn = "table.biom"
    taxonomy_fn = "taxonomy.tsv"
    axes = np.array(["counts", "tokens", "y", "encoder"])
    # # These are the UTF-8 encodings of A, C, T, G respectively
    # # lookup table converts utf-8 encodings to token
    # # tokens start at 1 to make room for pad token
    lookup_table = np.vectorize(map_unicode)

    def __init__(
        self,
        table: Union[str, Table] = None,
        shuffle: bool = False,
        epochs: int = 1000,
        batch_size: int = 8,
        max_bp: int = 150,
        seed=None,
    ):
        if table is not None:
            self.table = table

        self.shuffle = shuffle
        self.epochs = epochs
        self.samples_per_minibatch = batch_size

        self.batch_size = batch_size
        self.max_bp = max_bp
        self.seed = seed

        self.obs_ids = self.table.ids(axis="observation")

        print(f"Table shape: {self.table.shape}")
        self.sample_indices = np.arange(len(self.obs_ids))
        self.size = len(self.sample_indices)
        self.steps_per_epoch = self.size // self.batch_size
        self.table_data = self._create_table_data(self.table)

    @property
    def table(self) -> Table:
        if isinstance(self._table, str):
            self._table = load_table(self._table)
        return self._table

    @table.setter
    @add_lock
    def table(self, table: Union[str, Table]):
        if not isinstance(table, (str, Table)):
            tt = type(table)
            raise TypeError(f"Invalid table type. Expected file path or Table but recieve {tt}")

        if isinstance(table, str):
            if not os.path.exists(table):
                raise Exception(f"{table} is an invalid file path")

        self._table = table

    def _table_data(self, table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = table.copy()
        table = table.transpose()
        shape = table.shape
        coo = table.matrix_data.tocoo()
        (data, (row, col)) = (coo.data, coo.coords)
        return data, row, col, shape

    def _create_table_data(self, table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_ids = table.ids(axis="observation")
        sample_ids = table.ids()

        obs_encodings = None
        obs_encodings = np.array([[ord(char) for char in string] for string in obs_ids])
        obs_encodings = self.lookup_table(obs_encodings)

        table_data, row, col, shape = self._table_data(table)
        # only keep observations with count > 0
        table_mask = table_data > 0
        table_data = table_data[table_mask]
        row = row[table_mask]
        col = col[table_mask]

        return row, col, table_data, obs_encodings, sample_ids, obs_ids

    def _sample_data(self, samples: np.ndarray, table_data=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if table_data is None:
            table_data = self.table_data
        row, col, counts, obs_encodings, sample_ids, obs_ids = table_data

        return obs_encodings[samples]

    def _epoch_complete(self, processed):
        if processed < self.steps_per_epoch:
            return False
        return True

    def _minibatch_indices(self, minibatch, sample_indices):
        start = (minibatch * self.samples_per_minibatch) % len(sample_indices)
        end = (start + self.samples_per_minibatch) % len(sample_indices)
        if start > end:
            start = 0
            end = self.samples_per_minibatch
        return sample_indices[start:end]

    def _epoch_samples(self, table_data, sample_indices):
        if self.shuffle:
            print("shuffling...")
            np.random.shuffle(sample_indices)

        return table_data, sample_indices

    def _create_epoch_generator(self, include_seq_id, include_sample_ids):
        def generator():
            processed = 0
            table_data = self.table_data
            sample_indices = self.sample_indices
            for epoch in range(self.epochs):
                print(f"Finished epcoh: {epoch} processed {processed}")
                processed = 0
                minibatch = 0
                table_data, sample_indices = self._epoch_samples(table_data, sample_indices)

                def sample_data(minibatch):
                    samples = self._minibatch_indices(minibatch, sample_indices)
                    return self._sample_data(samples, table_data)

                while not self._epoch_complete(processed):
                    tokens = sample_data(minibatch)

                    table_output = tokens.astype(np.int32)

                    yield table_output
                    processed += 1
                    minibatch += 1

        return generator

    def get_data(self, include_seq_id=False, include_sample_ids=False):
        generator = self._create_epoch_generator(include_seq_id, include_sample_ids)

        output_sig = tf.TensorSpec(shape=[None, self.max_bp], dtype=tf.int32)
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_sig,
        )

        data_obj = {
            "dataset": dataset,
            "size": self.size,
            "steps_pre_epoch": self.steps_per_epoch,
        }
        return data_obj


if __name__ == "__main__":
    import numpy as np

    from aam.data_handlers import ASVGenerator

    ug = ASVGenerator(
        table="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom",
    )
    data = ug.get_data()
    for i, tokens in enumerate(data["dataset"]):
        print(tokens)
        break

    # data = ug.get_data_by_id(ug.rarefy_tables.ids()[:16])
    # for x, y in data["dataset"]:
    #     print(y)
    #     break
