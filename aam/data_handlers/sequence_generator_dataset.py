from __future__ import annotations

from typing import Optional, Union

import numpy as np
import tensorflow as tf
from biom import Table

from aam.data_handlers import SequenceDataset


class SequenceGeneratorDataset(SequenceDataset):
    def __init__(
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
        rare_freq: int = 10,
        epochs: int = 1000,
        num_tables: int = 1,
        gen_new_tables: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_token_per_sample = max_token_per_sample
        self.metadata_col = metadata_col
        self.is_categorical = is_categorical
        self.shift = shift
        self.scale = scale
        self.tax_level = tax_level
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.rarefy_depth = rarefy_depth
        self.rare_freq = rare_freq
        self.epochs = epochs
        self.num_tables = num_tables
        self.gen_new_tables = gen_new_tables

        if metadata_col is not None and self.metadata is None:
            raise Exception("No metadata present.")
        if tax_level is not None and self.taxonomy is None:
            raise Exception("No taxonomy present.")

        table = self._preprocess_data(rarefy_depth)

        level = None
        if self.taxonomy is not None and isinstance(tax_level, str):
            table, level = self._taxonomy_dataset(table, tax_level)

        shape = table.shape
        if self.metadata is not None and isinstance(metadata_col, str):
            table, target = self._metadata_dataset(
                table, metadata_col, is_categorical, shift, scale
            )
            if not np.array_equal(shape[0], table.shape[0]):
                raise Exception("Data out of alignment")

            table_ids = table.ids()
            target_ids = target.index.to_numpy()
            if not np.array_equal(table_ids, target_ids):
                raise Exception(
                    "Data out of alignment",
                    set(table).symmetric_difference(target_ids),
                )
            self.target = target.to_numpy().reshape((-1, 1))

        self.preprocessed_table = table
        self.obs_ids = self.preprocessed_table.ids(axis="observation")

        if level is not None:
            self.level = level.loc[self.obs_ids, :]

        print(f"creating {self.num_tables} table(s)...")
        self.rarefy_tables = [
            self.preprocessed_table.copy().subsample(self.rarefy_depth)
            for _ in range(self.num_tables)
        ]
        self.size = (
            self.preprocessed_table.shape[1] // self.batch_size
        ) * self.num_tables

    def _level_weights(self, table: Table) -> np.ndarray:
        all_ids = self.preprocessed_table.ids(axis="observation")
        level = self.level.loc[all_ids, ["token"]]

        # counts = table.pa(inplace=False).sum(axis="observation")
        counts = table.sum(axis="observation")
        table_obs = table.ids(axis="observation")
        level.loc[:, "counts"] = 0
        level.loc[table_obs, "counts"] = counts
        level_counts = level.groupby("token").agg("sum")

        level_counts = level_counts.to_numpy().reshape(-1)
        level_mask = level_counts > 0
        non_zero = np.sum(level_mask)
        level_counts[level_mask] = np.sum(level_counts) / (
            level_counts[level_mask] * non_zero
        )
        return np.pad(level_counts, (1, 0), constant_values=0)

    def _preprocess_data(self, rarefy_depth: int) -> Table:
        counts = self.table.sum(axis="sample")
        count_mask = counts >= rarefy_depth
        valid_samples = self.table.ids(axis="sample")[count_mask]
        table = self.table.filter(valid_samples, axis="sample", inplace=False)
        return table

    def _create_table_data(self, table, id):
        obs_ids = table.ids(axis="observation")
        obs_encodings = tf.cast(
            tf.strings.unicode_decode(obs_ids, "UTF-8"), dtype=tf.int64
        )
        obs_encodings = self.lookup_table.lookup(obs_encodings).numpy()

        table_data, row, col, shape = self._table_data(table)

        def _add_id(array: np.ndarray) -> np.ndarray:
            id_array = np.ones_like(array) * id
            return np.concatenate(
                (id_array.reshape((-1, 1)), array.reshape((-1, 1))), axis=1
            )

        return _add_id(row), col, table_data, obs_encodings, obs_ids

    def _sample_indx(self, sample):
        return sample[1]

    def _sample_id(self, sample):
        sample_indx = self._sample_indx(sample)
        return self.preprocessed_table.ids()[sample_indx]

    def _unique_samples(self, tables):
        samples = np.concatenate([t[0] for t in tables], axis=0)
        return np.unique(samples, axis=0)

    def _sample_data(
        self,
        sample: np.ndarray,
        tables: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        level_weights: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        table_indx = sample[0]
        row, col, table_data, obs_encodings, obs_ids = tables[table_indx]
        sample_indx = sample[1]
        target_data = self.target[sample_indx]

        sample_mask = row[:, 1] == sample_indx
        sample_counts = table_data[sample_mask]
        obs_indices = col[sample_mask]
        tokens = obs_encodings[obs_indices]

        sample_obs = obs_ids[obs_indices]
        tax_tokens = self.level.loc[sample_obs, "token"].to_numpy(dtype=np.int32)

        weights = level_weights[table_indx]
        weights = weights[tax_tokens]

        return (
            target_data,
            sample_counts,
            tokens,
            tax_tokens,
            weights,
        )

    def _epoch_complete(self, processed):
        if processed < self.size * self.batch_size:
            return False
        return True

    def _create_generator(self):
        def generator():
            table_data = [
                self._create_table_data(table, i)
                for i, table in enumerate(self.rarefy_tables)
            ]
            level_weights = [self._level_weights(table) for table in self.rarefy_tables]
            samples = self._unique_samples(table_data)
            samples_skipped = 0
            processed = 0

            for epoch in range(self.epochs):
                processed = 0

                if epoch > 0 and self.gen_new_tables:
                    print("generating new table...")
                    new_table = self.preprocessed_table.subsample(self.rarefy_depth)
                    table_data[-1] = self._create_table_data(
                        new_table, self.num_tables - 1
                    )
                    level_weights[-1] = self._level_weights(new_table)

                while not self._epoch_complete(processed):
                    samples = self._unique_samples(table_data)

                    if self.shuffle:
                        np.random.shuffle(samples)

                    for s in samples:
                        if self._epoch_complete(processed):
                            break

                        target, sample_counts, tokens, tax_tokens, weights = (
                            self._sample_data(s, table_data, level_weights)
                        )
                        if len(sample_counts) > self.max_token_per_sample:
                            samples_skipped += 1
                        else:
                            sorted_order = np.argsort(sample_counts)
                            sorted_order = sorted_order[::-1]
                            sorted_counts = sample_counts[sorted_order].reshape(-1, 1)
                            sorted_tokens = tokens[sorted_order]
                            sorted_tax_tokens = tax_tokens[sorted_order]

                            processed += 1

                            yield (
                                (
                                    sorted_tokens.astype(np.int32),
                                    sorted_counts.astype(np.int32),
                                ),
                                ((target, sorted_tax_tokens.astype(np.int32)), weights),
                            )

        return generator

    def get_data(self):
        generator = self._create_generator()
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=[None, 150], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
                ),
                (
                    (
                        tf.TensorSpec(shape=[1], dtype=tf.float32),
                        tf.TensorSpec(shape=[None], dtype=tf.int32),
                    ),
                    tf.TensorSpec(shape=[None], dtype=tf.float32),
                ),
            ),
        )
        dataset = dataset.padded_batch(
            self.batch_size,
            (
                ([None, 150], [None, 1]),
                (([1], [None]), [None]),
            ),
        ).prefetch(tf.data.AUTOTUNE)

        if not self.is_categorical:
            data_obj = {
                "dataset": dataset,
                "shift": self.shift,
                "scale": self.scale,
                "size": self.size,
            }
            if self.tax_level is not None:
                data_obj["num_tax_levels"] = max(self.level.loc[:, "token"]) + 1
            else:
                data_obj["num_tax_levels"] = None
        return data_obj


if __name__ == "__main__":
    import pandas as pd
    from biom import load_table

    table = load_table(
        "/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-small.biom"
    )
    metadata = pd.read_csv(
        "/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-healthy.txt",
        sep="\t",
        index_col=0,
    )
    sd = SequenceGeneratorDataset(
        150,
        "host_age",
        False,
        0.0,
        1.0,
        "Level 7",
        False,
        table=table,
        metadata=metadata,
        taxonomy="/home/kalen/aam-research-exam/research-exam/healty-age-regression/taxonomy.tsv",
    )
    samples = sd._create_table_data(sd.rarefy_tables[0], 1)
