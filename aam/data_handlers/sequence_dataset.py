from __future__ import annotations

import os
import shutil
import tempfile
from functools import wraps
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table
from biom.util import biom_open
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


class SequenceDataset:
    taxon_field = "Taxon"
    levels = [f"Level {i}" for i in range(1, 8)]
    level_attrs = {f"l{lvl[-1]}": lvl for lvl in levels}
    empty_level = ["d__", "p__", "c__", "o__", "f__", "g__", "s__"]
    table_fn = "table.biom"
    taxonomy_fn = "taxonomy.tsv"

    # These are the UTF-8 encodings of A, C, T, G respectively
    # lookup table converts utf-8 encodings to token
    # tokens start at 1 to make room for pad token
    key_val_init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([65, 67, 71, 84], dtype=tf.int64),
        values=tf.constant([1, 2, 3, 4], dtype=tf.int64),
    )
    lookup_table = tf.lookup.StaticVocabularyTable(key_val_init, num_oov_buckets=1)

    def __init__(
        self,
        table: Union[str, Table],
        metadata: Optional[Union[str, pd.DataFrame]] = None,
        taxonomy: Optional[Union[str, pd.DataFrame]] = None,
    ) -> SequenceDataset:
        self.table = table
        self.metadata = metadata
        self.taxonomy = taxonomy

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
            raise TypeError(
                f"Invalid table type. Expected file path or Table but recieve {tt}"
            )

        if isinstance(table, str):
            if not os.path.exists(table):
                raise Exception(f"{table} is an invalid file path")

        self._table = table

    @property
    def metadata(self) -> pd.DataFrame:
        if isinstance(self._metadata, str):
            self._metadata = pd.read_csv(self._metadata, sep="\t", index_col=0)
        return self._metadata

    @metadata.setter
    @add_lock
    def metadata(self, metadata: Union[str, pd.DataFrame]):
        if metadata is None:
            self._metadata = metadata
            return
        self._validate_dataframe(metadata)
        self._metadata = metadata

    @property
    def taxonomy(self) -> pd.DataFrame:
        if isinstance(self._taxonomy, str):
            self._taxonomy = pd.read_csv(self._taxonomy, sep="\t", index_col=0)
            if self.taxon_field not in self._taxonomy.columns:
                raise Exception("Invalid taxonomy: missing 'Taxon' field")
        if self._taxonomy is not None and not self._taxonomy_built:
            self._taxonomy[self.levels] = self._taxonomy[self.taxon_field].str.split(
                "; ", expand=True
            )
            self._taxonomy_built = True
        return self._taxonomy

    @taxonomy.setter
    @add_lock
    def taxonomy(self, taxonomy: Union[str, pd.DataFrame]):
        if hasattr(self, "_taxon_set"):
            raise Exception("Taxon already set")
        if taxonomy is None:
            self._taxonomy = taxonomy
            return

        self._taxonomy_built = False
        self._validate_dataframe(taxonomy)
        self._taxonomy = taxonomy
        self._taxon_set = True

    def _align_to_taxonomy(self, table, taxonomy):
        obs_id = np.intersect1d(table.ids(axis="observation"), taxonomy.index)
        table = table.filter(obs_id, axis="observation", inplace=False)
        taxonomy = taxonomy.loc[table.ids(axis="observation")]
        taxonomy = taxonomy.reindex(table.ids(axis="observation"))
        return table, taxonomy

    def _retrieve_level(self, level: str) -> pd.DataFrame:
        taxonomy = self.taxonomy
        if level not in self.levels:
            raise Exception(f"Invalid level: {level}")

        if taxonomy is None:
            raise Exception("Missing taxonomy")

        level_index = self.levels.index(level)
        level_taxonomy = taxonomy.loc[:, self.levels[: level_index + 1]]
        return level_taxonomy

    def level_classes(self, table: Table, level: str) -> pd.DataFrame:
        level_taxonomy = self._retrieve_level(level)
        table, level_taxonomy = self._align_to_taxonomy(table, level_taxonomy)
        level_taxonomy.loc[:, "class"] = level_taxonomy.loc[
            :, self.levels[: self.levels.index(level) + 1]
        ].agg("; ".join, axis=1)

        return table, level_taxonomy

    def total_seq_dropped(
        self, valid_mask: pd.Series, table: Table, return_ratio: bool = False
    ) -> float:
        dropped_ids = self.taxonomy.loc[~valid_mask, :].index
        dropped_table = table.filter(dropped_ids, axis="observation", inplace=False)
        dropped = dropped_table.sum()
        if return_ratio:
            return dropped / table.sum()
        return dropped

    def save(self, fp: str, overwrite: bool = False):
        if os.path.exists(fp):
            if not overwrite:
                raise Exception(f"{fp} already exists.")
            os.remove(fp)

        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = os.path.join(tmpdir, SequenceDataset.table_fn)
            with biom_open(table_path, "w") as f:
                self.table.to_hdf5(f, "SequenceDataset")

            if self.taxonomy is not None:
                taxonomy_path = os.path.join(tmpdir, SequenceDataset.taxonomy_fn)
                self.taxonomy.to_csv(taxonomy_path, sep="\t", index=True)

            shutil.make_archive(fp, "zip", tmpdir)
        os.rename(f"{fp}.zip", fp)

    @classmethod
    def load(cls, fp: str) -> SequenceDataset:
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.unpack_archive(fp, tmpdir, "zip")

            table_path = os.path.join(tmpdir, cls.table_fn)
            if not os.path.exists(table_path):
                raise Exception(f"{fp} is an invalid SequenceDataset object")
            table = load_table(table_path)

            taxonomy_path = os.path.join(tmpdir, cls.taxonomy_fn)
            if not os.path.exists(taxonomy_path):
                taxonomy = None
            else:
                taxonomy = pd.read_csv(taxonomy_path, sep="\t", index_col=0)
        return cls(table, taxonomy)

    def _table_data(self, table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = table.copy()
        table = table.transpose()
        shape = table.shape
        coo = table.matrix_data.tocoo()
        (data, (row, col)) = (coo.data, coo.coords)
        return data, row, col, shape

    def _table_dataset(self, table: Table) -> tf.data.Dataset:
        data, row, col, shape = self._table_data(table)
        indices = tf.concat(
            [tf.expand_dims(row, axis=1), tf.expand_dims(col, axis=1)], axis=1
        )
        table_data = tf.sparse.SparseTensor(
            indices=tf.cast(indices, dtype=tf.int64),
            values=data,
            dense_shape=shape,
        )
        table_data = tf.sparse.reorder(table_data)
        table_data = tf.data.Dataset.from_tensor_slices(table_data)
        return table_data

    def _metadata_dataset(
        self,
        table: Table,
        metadata_col: Optional[str] = None,
        is_categorical: Optional[bool] = None,
        shift: Optional[Union[str, float]] = None,
        scale: Union[str, float] = "minmax",
    ) -> tuple[Table, np.ndarray]:
        metadata = self.metadata
        if metadata_col not in metadata.columns:
            raise Exception(f"Invalid metadata column {metadata_col}")
        samp_ids = np.intersect1d(table.ids(axis="sample"), metadata.index)
        table = table.filter(samp_ids, axis="sample", inplace=False)
        metadata = metadata.loc[table.ids(), metadata_col]
        print(f"table shape: {table.shape}")
        print(f"metadata shape: {metadata.shape}")
        if not is_categorical:
            metadata = metadata.astype(np.float32)
            if not isinstance(scale, (str, float)):
                raise Exception("Invalid scale argument.")
            if shift is None and isinstance(scale, float):
                raise Exception("Invalid shift argument")

            if scale == "minmax":
                shift = np.min(metadata)
                scale = np.max(metadata) - shift
            elif scale == "standscale":
                shift = np.mean(metadata)
                scale = np.std(metadata)

            metadata = (metadata - shift) / scale
        metadata = metadata.reindex(table.ids())
        return table, metadata

    def _taxonomy_dataset(
        self, table: Table, tax_level: str
    ) -> tuple[Table, pd.Series]:
        table, level_tax = self.level_classes(table, tax_level)
        le = preprocessing.LabelEncoder()
        level_tax.loc[:, "token"] = le.fit_transform(level_tax["class"])
        level_tax.loc[:, "token"] += 1
        return table, level_tax

    def _process_dataset(
        self,
        obs_ids: np.ndarray,
        datasets: list[tf.data.Dataset],
        max_token_per_sample: int = 512,
        shuffle: bool = True,
        batch_size: int = 8,
        level: Optional[pd.Series] = None,
    ) -> tf.data.Dataset:
        obs_encodings = tf.cast(
            tf.strings.unicode_decode(obs_ids, "UTF-8"), dtype=tf.int64
        )
        if level is not None:
            level_encodings = tf.cast(
                np.squeeze(level.loc[obs_ids, "token"].to_numpy()), dtype=tf.int64
            )

        def apply_func(shuffle_buf=32, include_tax=False):
            def _inner(ds):
                def process_table(data, target_data):
                    sorted_order = tf.argsort(
                        data.values, axis=-1, direction="DESCENDING"
                    )

                    asv_indices = tf.reshape(data.indices, shape=[-1])
                    sorted_asv_indices = tf.gather(asv_indices, sorted_order)[
                        :max_token_per_sample
                    ]
                    counts = tf.gather(data.values, sorted_order)[:max_token_per_sample]
                    counts = tf.cast(counts, dtype=tf.int32)
                    counts = tf.expand_dims(counts, axis=-1)

                    encodings = tf.gather(obs_encodings, sorted_asv_indices)
                    tokens = self.lookup_table.lookup(encodings).to_tensor()
                    if not include_tax:
                        output = (tf.cast(tokens, dtype=tf.int32), counts), target_data
                    else:
                        level_tokens = tf.gather(level_encodings, sorted_asv_indices)
                        level_tokens = tf.expand_dims(level_tokens, axis=-1)
                        output = (
                            (tf.cast(tokens, dtype=tf.int32), counts),
                            (target_data, tf.cast(level_tokens, dtype=tf.int32)),
                        )
                    return output

                if shuffle:
                    ds = ds.shuffle(shuffle_buf)

                ds = ds.map(process_table, num_parallel_calls=tf.data.AUTOTUNE)
                if not include_tax:
                    ds = ds.padded_batch(
                        batch_size,
                        (([None, 150], [None, 1]), [1]),
                    )
                else:
                    ds = ds.padded_batch(
                        batch_size,
                        (
                            ([None, 150], [None, 1]),
                            ([1], [None, 1]),
                        ),
                    )

                return ds

            return _inner

        include_tax = level is not None
        dataset = tf.data.Dataset.zip(*datasets)
        return dataset.apply(apply_func(include_tax=include_tax)).prefetch(
            tf.data.AUTOTUNE
        )

    def get_data(
        self,
        max_token_per_sample: int = 512,
        metadata_col: Optional[str] = None,
        is_categorical: Optional[bool] = None,
        shift: Optional[Union[str, float]] = None,
        scale: Union[str, float] = "minmax",
        tax_level: Optional[str] = None,
        shuffle: bool = True,
        batch_size: int = 8,
        rarefy_depth: int = 5000,
    ):
        if metadata_col is not None and self.metadata is None:
            raise Exception("No metadata present.")
        if tax_level is not None and self.taxonomy is None:
            raise Exception("No taxonomy present.")

        table = self.table.copy()
        additional_data = []

        if self.taxonomy is not None and isinstance(tax_level, str):
            table, level = self._taxonomy_dataset(table, tax_level)

        num_obs = table.shape[0]
        if self.metadata is not None and isinstance(metadata_col, str):
            table, metadata = self._metadata_dataset(
                table, metadata_col, is_categorical, shift, scale
            )
            metadata = tf.data.Dataset.from_tensor_slices(metadata)
            additional_data.append(metadata)
            if num_obs != table.shape[0]:
                raise Exception("Data out of alignment")
        obs_ids = table.ids(axis="observation")

        table_data = self._table_dataset(table)
        dataset = self._process_dataset(
            obs_ids,
            [table_data] + additional_data,
            max_token_per_sample,
            shuffle,
            batch_size,
            level,
        )
        if not is_categorical:
            data_obj = {
                "dataset": dataset,
                "shift": shift,
                "scale": scale,
            }
            if tax_level is not None:
                data_obj["num_tax_levels"] = max(level.loc[:, "token"]) + 1
            else:
                data_obj["num_tax_levels"] = None
        return data_obj
