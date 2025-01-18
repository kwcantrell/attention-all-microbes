from __future__ import annotations

import os
from functools import wraps
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table

from aam.data_handlers.asv_generator import tokenize_asv


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


# Unicode mapping dictionary
mapping = {65: 1, 67: 2, 71: 3, 84: 4}  # Maps Unicode numbers to specific values


# Create the mapping function
def map_unicode(val):
    return mapping.get(val, 0)  # Return 0 if the value is not in the mapping


class GeneratorDataset(tf.keras.utils.Sequence):
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
        seed=None,
    ):
        if table is not None:
            self.table = table

        self.metadata_column = metadata_column
        self.is_categorical = is_categorical
        self.include_sample_weight = is_categorical
        self.shift = shift
        self.scale = scale

        if table is not None:
            self.metadata = metadata

        self.max_token_per_sample = max_token_per_sample
        self.shuffle = shuffle
        self.rarefy_depth = rarefy_depth
        self.epochs = epochs
        self.gen_new_tables = gen_new_tables
        self.samples_per_minibatch = batch_size

        self.batch_size = batch_size
        self.max_bp = max_bp
        self.is_16S = is_16S
        self.seed = seed
        self.gen_new_table_frequency = gen_new_table_frequency
        self.epochs_since_last_table = 0

        if table is not None:
            self.preprocessed_table = self.table
            self.on_epoch_end()
            self.encoder_target = None
            self.encoder_dtype = None
            self.encoder_output_type = None

    def create_rarefied_table(self, table):
        rarefied_table = table.subsample(self.rarefy_depth, seed=self.seed)
        sample_mask = rarefied_table.pa(inplace=False).sum(axis="sample") <= self.max_token_per_sample
        return rarefied_table, sample_mask

    def _table_data(self, table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = table.copy()
        table = table.transpose()
        shape = table.shape
        coo = table.matrix_data.tocoo()
        (data, (row, col)) = (coo.data, coo.coords)
        return data, row, col, shape

    def _create_table_data(self, table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_encodings = table.ids(axis="observation")
        s_ids = table.ids(axis="sample")

        table_counts, row, col, _ = self._table_data(table)

        # only keep observations with count > 0
        table_mask = table_counts > 0
        table_counts = table_counts[table_mask]
        row = row[table_mask]
        col = col[table_mask]

        return row, col, table_counts, obs_encodings, s_ids

    def _create_y_data(self, table: Table) -> pd.Series:
        if self.metadata_column is None or self.metadata is None:
            return None

        table_ids = table.ids()
        return self.metadata.loc[table_ids]

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size

        samples = self.sample_indices[start:end]
        (batch_counts, counts, tokens, indices, y_output, encoder_out, ob_ids, s_ids) = self._sample_data(samples)

        if counts is not None:
            table_output = (
                batch_counts.astype(np.int32),
                tokens,
                indices.astype(np.int32),
                counts.astype(np.int32),
            )

        output = None
        if y_output is not None:
            output = y_output.astype(np.float32)

        if encoder_out is not None:
            if isinstance(encoder_out, tuple):
                encoder_out = tuple([o.astype(t) for o, t in zip(encoder_out, self.encoder_dtype)])
            else:
                encoder_out = encoder_out.astype(self.encoder_dtype)

            if output is not None:
                output = (output, encoder_out)
            else:
                output = encoder_out

        if output is not None:
            return (table_output, output)
        else:
            return table_output

    def _sample_data(self, samples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row, col, counts, obs_encodings, sample_ids = self.table_data

        if max(samples) >= len(sample_ids):
            raise Exception(f"\tsample_indices exceed max {len(sample_ids)}. samples {samples}...")
        s_ids = [sample_ids[s] for s in samples]

        samples = samples.reshape((-1, 1))
        row = row.reshape((1, -1))
        batch_mask = samples == row
        batch_counts = np.sum(batch_mask.astype(np.int32), axis=-1)
        batch_mask = np.logical_or.reduce(batch_mask, axis=0)

        s_counts = counts[batch_mask]

        s_obj_ids = col[batch_mask]
        if self.is_16S:
            unique_obj, obj_indices = np.unique(s_obj_ids, return_inverse=True)
            s_tokens = obs_encodings[unique_obj]
        else:
            s_tokens = self.gotu_tokens(s_obj_ids)
        s_max_token = np.max(batch_counts)

        if s_max_token > self.max_token_per_sample:
            print(f"\tskipping group due to exceeding token limit {s_max_token}...")
            return None, None, None, None, None, None

        y_output = self._y_output(self.y_data, s_ids)
        encoder_output = self._encoder_output(self.encoder_target, s_ids, s_obj_ids)

        return (
            batch_counts,
            s_counts.reshape((-1, 1)),
            s_tokens,
            obj_indices,
            y_output,
            encoder_output,
            s_obj_ids,
            s_ids,
        )

    def _create_table(self):
        self.rarefy_table, self.sample_mask = self.create_rarefied_table(self.preprocessed_table)

        print(f"Postrarefaction Table shape: {self.rarefy_table.shape}")
        self.sample_indices = np.arange(len(self.rarefy_table.ids()))

        if not hasattr(self, "steps_per_epoch"):
            self.size = len(self.sample_indices)
            self.steps_per_epoch = self.size // self.batch_size

        self.sample_indices = self.sample_indices[self.sample_mask]
        fill_out = (self.size // len(self.sample_indices)) + 1
        if fill_out > 0:
            self.sample_indices = np.repeat([self.sample_indices], repeats=fill_out, axis=0).reshape((-1))

        self.table_data = self._create_table_data(self.rarefy_table)
        self.y_data = self._create_y_data(self.rarefy_table)
        self.epochs_since_last_table = 0

    def on_epoch_end(self):
        print("creating table...")
        print(f"Prerafaction Table shape: {self.preprocessed_table.shape}")

        if self.epochs_since_last_table >= self.gen_new_table_frequency or not hasattr(self, "steps_per_epoch"):
            self._create_table()

        self.epochs_since_last_table += 1

        if self.shuffle:
            np.random.shuffle(self.sample_indices)

    def _validate_dataframe(self, df: pd.DataFrame):
        if isinstance(df, str):
            if not os.path.exists(df):
                raise TypeError(f"Invalid path: {df}")
        elif not isinstance(df, pd.DataFrame):
            raise TypeError("Excepted a file path or DataFrame")

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

    @property
    def metadata(self) -> pd.Series:
        return self._metadata

    @metadata.setter
    @add_lock
    def metadata(self, metadata: Union[str, pd.DataFrame]):
        if metadata is None:
            self._metadata = metadata
            return
        self._validate_dataframe(metadata)
        if isinstance(metadata, str):
            metadata = pd.read_csv(metadata, sep="\t", index_col=0, dtype={0: str})

        if self.metadata_column not in metadata.columns:
            raise Exception(f"Invalid metadata column {self.metadata_column}")

        samp_ids = np.intersect1d(self.table.ids(axis="sample"), metadata.index)
        table = self.table.filter(samp_ids, axis="sample", inplace=False)
        metadata = metadata.loc[table.ids(), self.metadata_column]
        print(f"table shape: {table.shape}")
        print(f"metadata shape: {metadata.shape}")
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
        self._metadata = metadata.reindex(table.ids())
        self._table = table

    def _create_encoder_target(self, table: Table) -> None:
        return None

    def _encoder_output(self, encoder_target, sample_ids, obs_ids):
        return None

    def _y_output(self, y_data: Optional[pd.Series], sample_ids: Iterable[str]) -> np.ndarray:
        if y_data is None:
            return None

        if not (y_data, pd.Series):
            raise Exception(f"Invalid y_data object: {type(y_data)}")

        if not self.is_categorical:
            return y_data.loc[sample_ids].to_numpy().reshape(-1, 1)

        return y_data.loc[sample_ids].to_numpy().reshape(-1, 1)


if __name__ == "__main__":
    ug = GeneratorDataset(
        table="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom",
        metadata="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-healthy.txt",
        metadata_column="host_age",
        scale="minmax",
        gen_new_tables=True,
        max_token_per_sample=100,
        batch_size=4,
    )

    for i, (x, y) in enumerate(ug):
        print(x, y)
