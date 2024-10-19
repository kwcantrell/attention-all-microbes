from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from biom import Table
from biom.util import biom_open
from unifrac import unweighted

from aam.data_handlers import SequenceDataset


class UniFracGeneratorDataset(SequenceDataset):
    def __init__(
        self,
        max_token_per_sample: int = 512,
        shuffle: bool = False,
        rarefy_depth: int = 5000,
        epochs: int = 1000,
        num_tables: int = 1,
        gen_new_tables: bool = False,
        tree_path: str = None,
        samples_per_epoch: int = 100,
        batch_size: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_token_per_sample = max_token_per_sample
        self.shuffle = shuffle
        self.rarefy_depth = rarefy_depth
        self.epochs = epochs
        self.num_tables = num_tables
        self.gen_new_tables = gen_new_tables
        self.tree_path = tree_path
        self.samples_per_epoch = samples_per_epoch
        self.samples_per_minibatch = batch_size
        if batch_size % 2 != 0:
            raise Exception("Batch size must be multiple of 2")
        self.batch_size = batch_size

        self.preprocessed_table = self.table
        self.obs_ids = self.preprocessed_table.ids(axis="observation")

        print(f"creating {self.num_tables} table(s)...")
        self.random = np.random.random(1)[0]
        self.rarefy_tables = self.preprocessed_table.subsample(self.rarefy_depth)
        print(f"Table shape: {self.rarefy_tables.shape}")
        self.sample_indices = np.arange(len(self.rarefy_tables.ids()))
        np.random.shuffle(self.sample_indices)

        self.size = len(self.sample_indices)
        self.steps_per_epoch = self.size // self.batch_size
        self.table_data = self._create_table_data(self.rarefy_tables)
        self.distances = self._unifrac_distances(self.rarefy_tables, 0)

    def _create_table_data(self, table):
        obs_ids = table.ids(axis="observation")
        sample_ids = table.ids()

        obs_encodings = tf.cast(
            tf.strings.unicode_decode(obs_ids, "UTF-8"), dtype=tf.int64
        )
        obs_encodings = self.lookup_table.lookup(obs_encodings).numpy()

        table_data, row, col, shape = self._table_data(table)

        return row, col, table_data, obs_encodings, sample_ids

    def _sample_data(
        self, samples: np.ndarray, table_data=None, distances=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if table_data is None:
            table_data = self.table_data
        row, col, counts, obs_encodings, sample_ids = table_data

        if max(samples) >= len(sample_ids):
            return None, None, None

        def _s_info(s):
            s_mask = row == s
            s_counts = counts[s_mask]
            s_obs_indices = col[s_mask]
            s_tokens = obs_encodings[s_obs_indices]
            sorted_order = np.argsort(s_counts)
            sorted_order = sorted_order[::-1]
            s_counts = s_counts[sorted_order].reshape(-1, 1)
            s_tokens = s_tokens[sorted_order]
            return s_counts, s_tokens

        s_data = [_s_info(s) for s in samples]
        s_counts = [c for c, _ in s_data]
        s_tokens = [t for _, t in s_data]
        s_max_token = max([len(t) for t in s_tokens])
        if s_max_token > self.max_token_per_sample:
            return None, None, None

        s_ids = [sample_ids[s] for s in samples]

        if distances is None:
            distances = self.distances
        distance = distances.filter(s_ids).data

        return (s_counts, s_tokens, distance)

    def _unifrac_distances(self, table: Table, index: int):
        temp_path = f"/tmp/temp{self.random}{index}.biom"
        with biom_open(temp_path, "w") as f:
            table.to_hdf5(f, "aam")
        distances = unweighted(temp_path, self.tree_path)
        os.remove(temp_path)
        return distances

    def _epoch_complete(self, processed):
        if processed < self.steps_per_epoch:
            return False
        return True

    def _minibatch_indices(self, minibatch, sample_indices):
        start = (minibatch * self.samples_per_minibatch) % len(sample_indices)
        end = (start + self.samples_per_minibatch) % len(sample_indices)
        if start > end:
            if self.shuffle:
                print("shuffling...")
                np.random.shuffle(sample_indices)
            start = 0
            end = self.samples_per_minibatch
        return sample_indices[start:end]

    def _epoch_samples(self, epoch, old_table_data, old_distances, old_indices):
        if self.gen_new_tables and epoch > 0:
            print(f"epcoh {epoch}: generating new table...")
            rarefy_tables = self.preprocessed_table.subsample(self.rarefy_depth)
            table_data = self._create_table_data(rarefy_tables)
            distances = self._unifrac_distances(rarefy_tables, 0)
            sample_indices = np.arange(len(rarefy_tables.ids()))
        else:
            table_data = old_table_data
            distances = old_distances
            sample_indices = old_indices

        if self.shuffle:
            print("shuffling...")
            np.random.shuffle(sample_indices)

        return table_data, distances, sample_indices

    def _create_epoch_generator(self):
        def generator():
            processed = 0
            table_data = self.table_data
            distances = self.distances
            sample_indices = self.sample_indices
            for epoch in range(self.epochs):
                print(f"Finished epcoh: {epoch} processed {processed}")
                processed = 0
                minibatch = 0
                table_data, distances, sample_indices = self._epoch_samples(
                    epoch, table_data, distances, sample_indices
                )

                def sample_data(minibatch):
                    samples = self._minibatch_indices(minibatch, sample_indices)
                    return self._sample_data(samples, table_data, distances)

                while not self._epoch_complete(processed):
                    counts, tokens, distance = sample_data(minibatch)

                    if counts is not None:
                        max_len = max([len(c) for c in counts])
                        padded_counts = np.array(
                            [np.pad(c, [[0, max_len - len(c)], [0, 0]]) for c in counts]
                        )
                        padded_tokens = np.array(
                            [np.pad(t, [[0, max_len - len(t)], [0, 0]]) for t in tokens]
                        )

                        processed += 1
                        yield (
                            (
                                padded_tokens.astype(np.int32),
                                padded_counts.astype(np.int32),
                            ),
                            distance.astype(np.float32),
                        )
                    minibatch += 1

        return generator

    def _create_sample_generator(self, samples):
        table = self.rarefy_tables.filter(samples, inplace=False)
        table_data = self._create_table_data(table)
        distances = self._unifrac_distances(table, 0)
        s_indices = np.arange(len(samples))

        def generator():
            batch_samples = set()
            processed = 0
            for s in s_indices:
                batch_samples = [s]

                counts, tokens, distance = self._sample_data(batch_samples)

                if counts is not None:
                    max_len = max([len(c) for c in counts])
                    padded_counts = np.array(
                        [np.pad(c, [[0, max_len - len(c)], [0, 0]]) for c in counts]
                    )
                    padded_tokens = np.array(
                        [np.pad(t, [[0, max_len - len(t)], [0, 0]]) for t in tokens]
                    )

                    processed += 1
                    yield (
                        (
                            padded_tokens.astype(np.int32),
                            padded_counts.astype(np.int32),
                        ),
                        distance.astype(np.float32),
                    )

        return generator

    def get_data(self):
        generator = self._create_epoch_generator()
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=[self.batch_size, None, 150], dtype=tf.int32),
                    tf.TensorSpec(shape=[self.batch_size, None, 1], dtype=tf.int32),
                ),
                tf.TensorSpec(
                    shape=([self.batch_size, self.batch_size]), dtype=tf.float32
                ),
            ),
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        data_obj = {
            "dataset": dataset,
            "size": self.size,
            "steps_pre_epoch": self.steps_per_epoch,
        }
        return data_obj

    def get_data_by_id(self, sample_ids):
        intersect = np.intersect1d(sample_ids, self.rarefy_tables.ids())
        if len(intersect) != len(sample_ids):
            raise Exception(f"Invalid ids: {intersect}")


if __name__ == "__main__":
    import tensorflow as tf

    from aam.data_handlers import UniFracGeneratorDataset

    ug = UniFracGeneratorDataset(
        table="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom",
        tree_path="/home/kalen/aam-research-exam/research-exam/agp/data/agp-aligned.nwk",
        gen_new_tables=True,
    )
    data = ug.get_data()
    for i, (x, y) in enumerate(data["dataset"]):
        continue
