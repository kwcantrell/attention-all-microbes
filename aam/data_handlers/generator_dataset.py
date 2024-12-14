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


class GeneratorDataset:
    table_fn = "table.biom"
    taxonomy_fn = "taxonomy.tsv"
    axes = np.array(["counts", "tokens", "y", "encoder"])
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
        repeat=1,
        seed=None,
    ):
        table, metadata
        self.table = table

        self.metadata_column = metadata_column
        self.is_categorical = is_categorical
        self.include_sample_weight = is_categorical
        self.shift = shift
        self.scale = scale
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
        self.repeat = repeat

        self.preprocessed_table = self.table
        self.obs_ids = self.preprocessed_table.ids(axis="observation")
        self.seed = seed

        print("creating table...")
        self.rarefy_table, self.sample_mask = self.create_rarefied_table(
            self.preprocessed_table
        )

        print(f"Table shape: {self.rarefy_table.shape}")
        self.sample_indices = np.arange(len(self.rarefy_table.ids()))
        self.size = len(self.sample_indices)
        self.sample_indices = self.sample_indices[self.sample_mask]
        self.steps_per_epoch = (self.size // self.batch_size) * self.repeat
        self.table_data = self._create_table_data(self.rarefy_table)
        self.y_data = self._create_y_data(self.rarefy_table)
        self.encoder_target = None
        self.encoder_dtype = None
        self.encoder_output_type = None

    def create_rarefied_table(self, table):
        rarefied_table = table.subsample(self.rarefy_depth, seed=self.seed)
        sample_mask = (
            rarefied_table.pa(inplace=False).sum(axis="sample")
            <= self.max_token_per_sample
        )
        return rarefied_table, sample_mask

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

    def _table_data(self, table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = table.copy()
        table = table.transpose()
        shape = table.shape
        coo = table.matrix_data.tocoo()
        (data, (row, col)) = (coo.data, coo.coords)
        return data, row, col, shape

    def _create_encoder_target(self, table: Table) -> None:
        return None

    def _create_y_data(self, table: Table) -> pd.Series:
        if self.metadata_column is None or self.metadata is None:
            return None

        table_ids = table.ids()
        return self.metadata.loc[table_ids]

    def _encoder_output(self, encoder_target, sample_ids, obs_ids):
        return None

    def _y_output(
        self, y_data: Optional[pd.Series], sample_ids: Iterable[str]
    ) -> np.ndarray:
        if y_data is None:
            return None

        if not (y_data, pd.Series):
            raise Exception(f"Invalid y_data object: {type(y_data)}")

        if not self.is_categorical:
            return y_data.loc[sample_ids].to_numpy().reshape(-1, 1)

        return y_data.loc[sample_ids].to_numpy().reshape(-1, 1)

    def _create_table_data(
        self, table: Table
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_ids = table.ids(axis="observation")
        sample_ids = table.ids()

        obs_encodings = None
        if self.is_16S:
            obs_encodings = tf.cast(
                tf.strings.unicode_decode(obs_ids, "UTF-8"), dtype=tf.int64
            )
            obs_encodings = self.lookup_table.lookup(obs_encodings).numpy()

        table_data, row, col, shape = self._table_data(table)
        # only keep observations with count > 0
        table_mask = table_data > 0
        table_data = table_data[table_mask]
        row = row[table_mask]
        col = col[table_mask]

        return row, col, table_data, obs_encodings, sample_ids, obs_ids

    def _sample_data(
        self, samples: np.ndarray, table_data=None, y_data=None, encoder_target=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if table_data is None:
            table_data = self.table_data
        row, col, counts, obs_encodings, sample_ids, obs_ids = table_data

        if max(samples) >= len(sample_ids):
            raise Exception(
                f"\tsample_indices exceed max {len(sample_ids)}. samples {samples}..."
            )
        s_ids = [sample_ids[s] for s in samples]

        samples = samples.reshape((-1, 1))
        row = row.reshape((1, -1))
        batch_mask = samples == row
        batch_counts = np.sum(batch_mask.astype(np.int32), axis=-1)
        batch_mask = np.logical_or.reduce(batch_mask, axis=0)

        s_counts = counts[batch_mask]

        s_obj_ids = col[batch_mask]
        if self.is_16S:
            # asv_unique_obj, asv_obj_indices = np.unique(asv_s_obs, return_inverse=True)
            # asv_s_tokens = asv_obs_encodings[asv_unique_obj]
            unique_obj, obj_indices = np.unique(s_obj_ids, return_inverse=True)
            s_tokens = obs_encodings[unique_obj]
        else:
            s_tokens = self.gotu_tokens(s_obj_ids)
        s_max_token = np.max(batch_counts)

        if s_max_token > self.max_token_per_sample:
            print(f"\tskipping group due to exceeding token limit {s_max_token}...")
            return None, None, None, None, None, None

        if y_data is None:
            y_data = self.y_data
        y_output = self._y_output(y_data, s_ids)

        if encoder_target is None:
            encoder_target = self.encoder_target
        encoder_output = self._encoder_output(encoder_target, s_ids, s_obj_ids)

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

    def _epoch_samples(
        self,
        epoch,
        table_data,
        y_data,
        encoder_target,
        sample_mask,
        sample_indices,
    ):
        if self.gen_new_tables and epoch > 0:
            print(f"epcoh {epoch}: generating new table...")
            rarefy_table, sample_mask = self.create_rarefied_table(
                self.preprocessed_table
            )
            table_data = self._create_table_data(rarefy_table)
            y_data = self._create_y_data(rarefy_table)
            encoder_target = self._create_encoder_target(rarefy_table)
            sample_indices = np.arange(len(rarefy_table.ids()))[sample_mask]

        if self.shuffle:
            print("shuffling...")
            np.random.shuffle(sample_indices)

        return table_data, y_data, encoder_target, sample_mask, sample_indices

    def _create_epoch_generator(self, include_seq_id, include_sample_ids):
        def generator():
            processed = 0
            table_data = self.table_data
            y_data = self.y_data
            encoder_target = self.encoder_target
            sample_mask = self.sample_mask
            sample_indices = self.sample_indices
            for epoch in range(self.epochs):
                print(f"Finished epcoh: {epoch} processed {processed}")
                processed = 0
                minibatch = 0
                table_data, y_data, encoder_target, sample_mask, sample_indices = (
                    self._epoch_samples(
                        epoch,
                        table_data,
                        y_data,
                        encoder_target,
                        sample_mask,
                        sample_indices,
                    )
                )

                def sample_data(minibatch):
                    samples = self._minibatch_indices(minibatch, sample_indices)
                    return self._sample_data(
                        samples, table_data, y_data, encoder_target
                    )

                while not self._epoch_complete(processed):
                    (
                        batch_counts,
                        counts,
                        tokens,
                        indices,
                        y_output,
                        encoder_out,
                        ob_ids,
                        s_ids,
                    ) = sample_data(minibatch)

                    if counts is not None:
                        processed += 1
                        table_output = (
                            batch_counts.astype(np.int32),
                            tokens.astype(np.int32),
                            indices.astype(np.int32),
                            counts.astype(np.int32),
                        )

                        output = None
                        if y_output is not None:
                            output = y_output.astype(np.float32)

                        if encoder_out is not None:
                            if isinstance(encoder_out, tuple):
                                encoder_out = tuple(
                                    [
                                        o.astype(t)
                                        for o, t in zip(encoder_out, self.encoder_dtype)
                                    ]
                                )
                            else:
                                encoder_out = encoder_out.astype(self.encoder_dtype)

                            if output is not None:
                                output = (
                                    output,
                                    encoder_out,
                                )
                            else:
                                output = encoder_out

                        if include_seq_id:
                            output = (
                                *output,
                                ob_ids,
                            )
                        if include_sample_ids:
                            output = (*output, s_ids)

                        if output is not None:
                            yield (table_output, output)
                        else:
                            yield table_output
                    minibatch += 1

        return generator

    def _create_sample_generator(self, samples):
        table = self.rarefy_table.filter(samples, inplace=False)
        table_data = self._create_table_data(table)
        enocder_target = self._create_encoder_target(table)
        sample_indices = np.arange(len(table.ids()))
        sample_ids = table_data[-1]

        def generator():
            for s in range(0, len(sample_indices)):
                batch_samples = [sample_indices[s]]
                counts, tokens, encoder_out = self._sample_data(
                    batch_samples, table_data, enocder_target
                )

                if counts is not None:
                    sample_id = sample_ids[sample_indices[s]]
                    yield (
                        (
                            tokens[0].astype(np.int32),
                            counts[0].astype(np.int32),
                        ),
                        sample_id,
                    )

        return generator

    def get_data(self, include_seq_id=False, include_sample_ids=False):
        generator = self._create_epoch_generator(include_seq_id, include_sample_ids)

        if self.is_16S:
            output_sig = (
                tf.TensorSpec(shape=[self.batch_size], dtype=tf.int32),
                tf.TensorSpec(shape=[None, self.max_bp], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            )
        else:
            output_sig = (
                tf.TensorSpec(shape=[self.batch_size, None, 1], dtype=tf.int32),
                tf.TensorSpec(shape=[self.batch_size, None, 1], dtype=tf.int32),
            )

        y_output_sig = None
        if self.y_data is not None:
            y_output_sig = tf.TensorSpec(shape=[self.batch_size, 1], dtype=tf.float32)

        if self.encoder_target is not None:
            if y_output_sig is None:
                y_output_sig = self.encoder_output_type
            else:
                y_output_sig = (y_output_sig, self.encoder_output_type)

        if y_output_sig is not None:
            if include_seq_id:
                y_output_sig = (
                    *y_output_sig,
                    tf.TensorSpec(
                        shape=(self.batch_size, None), dtype=tf.string, name=None
                    ),
                )
            if include_sample_ids:
                y_output_sig = (
                    *y_output_sig,
                    tf.TensorSpec(shape=(self.batch_size,), dtype=tf.string, name=None),
                )
            output_sig = (output_sig, y_output_sig)
        class_weights = None
        if self.is_categorical:
            counts = self._metadata.value_counts().to_dict()
            counts[0.0] = 0
            class_weights = [0] * len(counts)
            for k, v in counts.items():
                class_weights[int(k)] = 1 / np.sqrt(v)
            class_weights[0] = 0
        print(class_weights)
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_sig,
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        data_obj = {
            "dataset": dataset,
            "shift": self.shift,
            "scale": self.scale,
            "size": self.size,
            "steps_pre_epoch": self.steps_per_epoch,
            "class_weights": class_weights,
        }
        return data_obj

    def get_data_by_id(
        self, samples: np.ndarray[str], axis: Union[str, tuple[str]] = None
    ):
        if isinstance(axis, str):
            axis = [axis]
        _, sample_indices = _matching_sample_indices(samples, self.table_data[-2])
        mask, _ = _matching_sample_indices(axis, self.axes)
        sample_data = self._sample_data(
            sample_indices, self.table_data, self.y_data, self.encoder_target
        )
        return [d for d, m in zip(sample_data, mask) if m]
