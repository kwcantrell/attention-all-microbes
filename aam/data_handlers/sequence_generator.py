from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from bp import parse_newick, to_skbio_treenode
from sklearn import preprocessing


class SequenceGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        sequences,
        shuffle: bool = False,
        epochs: int = 1000,
        sequence_batch_size: int = 8,
        seed=None,
        return_asv_ids=False,
        drop_remainder=True,
    ):
        self.sequences = sequences
        self.return_asv_ids = return_asv_ids
        self.drop_remainder = drop_remainder

        # extracting ASV tokens
        # step 1: find which nodes represent 150bp ASVs
        lookup = {
            "a": 1,
            "c": 2,
            "g": 3,
            "t": 4,
        }

        def _get_encodings(sequence):
            return [lookup[c] for c in sequence.lower()]

        self.obs_encodings = np.vstack(
            [_get_encodings(sequence) for sequence in self.sequences]
        )

        self.shuffle = shuffle
        self.epochs = epochs
        self.samples_per_minibatch = sequence_batch_size
        self.seed = seed

        self.size = len(self.obs_encodings)
        self.steps_per_epoch = max(self.size // self.samples_per_minibatch, 1)

        if (
            not self.drop_remainder
            and self.steps_per_epoch * self.samples_per_minibatch < self.size
        ):
            self.steps_per_epoch += 1

        self.sample_indices = np.arange(len(self.obs_encodings), dtype=np.int32)
        self.on_epoch_end()

        print("Number of sequences:", self.size)

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        start = idx * self.samples_per_minibatch
        end = start + self.samples_per_minibatch
        samples = self.sample_indices[start:end]
        return self._sample_data(samples)

    def on_epoch_end(self):
        print("Epoch finished")
        print("Preparing next epoch")
        if self.shuffle:
            np.random.shuffle(self.sample_indices)

    def _sample_data(
        self, samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = self.obs_encodings[samples]

        if self.return_asv_ids:
            return self.obs_encodings[samples], self.sequences[samples]

        return self.obs_encodings[samples]


if __name__ == "__main__":
    import numpy as np
    from biom import load_table

    table = load_table(
        "/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom"
    )
    ug = SequenceGenerator(
        sequences=table.ids(axis="observation"),
        sequence_batch_size=4,
        shuffle=False,
        return_asv_ids=True,
    )
    # model = tf.keras.models.load_model(
    #     "/home/kalen/aam-research-exam/research-exam/healty-age-regression/asv-encoder-tax-v4/model.keras",
    #     compile=False,
    # )
    print(ug[0])
