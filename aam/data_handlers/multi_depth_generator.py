from __future__ import annotations

import os
from functools import wraps
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table

from aam.data_handlers.asv_generator import tokenize_asv
from aam.data_handlers.unifrac_generator import UniFracGenerator


class MultiDepthGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        table: Union[str, Table],
        sample_depths: List[int],
        tree_path=None,
        unifrac_metric="unifrac",
        gen_new_table_frequency=3,
        batch_size=4,
        shuffle=False,
        **kwargs,
    ):
        if isinstance(table, str):
            table = load_table(table)

        kwargs["tree_path"] = tree_path
        kwargs["unifrac_metric"] = unifrac_metric
        self.generators = [
            UniFracGenerator(table=table, rarefy_depth=depth, shuffle=False, batch_size=batch_size, **kwargs)
            for depth in sample_depths
        ]

        self.batch_size = batch_size
        self.steps_per_epoch = min([g.steps_per_epoch for g in self.generators])
        self.size = self.batch_size * self.steps_per_epoch
        self.shuffle = shuffle
        self.gen_new_table_frequency = gen_new_table_frequency
        self.epochs_since_last_table = 0
        self.update_sample_indices()
        self.on_epoch_end()

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size

        sample_indices = self.sample_indices[start:end]
        sample_ids = self.common_ids[sample_indices]

        def _samples(sample_ids, gen):
            sample_mask = np.expand_dims(sample_ids, axis=-1) == gen.rarefy_table.ids()
            _sample_indices = np.argwhere(sample_mask)[:, -1]
            return gen._sample_data(_sample_indices)

        outputs = [_samples(sample_ids, gen) for gen in self.generators]
        combined_outputs = self._sample_data(outputs)

        (batch_counts, counts, tokens, indices, y_output, encoder_out, ob_ids, s_ids) = combined_outputs
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
                    encoder_out = tuple([o.astype(t) for o, t in zip(encoder_out, self.generators[0].encoder_dtype)])
                else:
                    encoder_out = encoder_out.astype(self.generators[0].encoder_dtype)

                if output is not None:
                    output = (output, encoder_out)
                else:
                    output = encoder_out

            if output is not None:
                return (table_output, output)
            else:
                return table_output

    def update_sample_indices(self):
        sample_ids = [g.rarefy_table.ids()[g.sample_mask] for g in self.generators]

        common_ids = set(sample_ids[0])
        for s_ids in sample_ids[1:]:
            common_ids = common_ids.intersection(s_ids)
        self.common_ids = np.array(list(common_ids))
        self.sample_indices = np.arange(len(self.common_ids))

        fill_out = (self.size // len(self.sample_indices)) + 1

        if self.shuffle:
            np.random.shuffle(self.sample_indices)

        if fill_out > 0:
            self.sample_indices = np.repeat([self.sample_indices], repeats=fill_out, axis=0).reshape((-1))

    def on_epoch_end(self):
        if self.epochs_since_last_table >= self.gen_new_table_frequency:
            for g in self.generators:
                g._create_table()
                self.epochs_since_last_table = 0
                self.update_sample_indices()

        self.epochs_since_last_table += 1

    def _sample_data(self, outputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_counts, counts, tokens, indices, y_output, encoder_out, ob_ids, s_ids = [], [], [], [], [], [], [], []
        shift = 0
        for bc, c, t, ind, yo, eo, oi, si in outputs:
            batch_counts.append(bc)
            counts.append(c.reshape(-1))
            tokens.append(t)
            indices.append(ind + shift)
            y_output.append(yo)
            encoder_out.append(eo)
            ob_ids.append(oi)
            s_ids.append(si)
            shift = np.max(ind + shift + 1)
        batch_counts = np.concatenate(batch_counts)
        counts = np.concatenate(counts)
        tokens = np.concatenate(tokens)
        indices = np.concatenate(indices)
        y_output = np.concatenate(y_output)
        encoder_out = np.concatenate(encoder_out)
        ob_ids = np.concatenate(ob_ids)
        s_ids = np.concatenate(s_ids)

        unique_t, unique_ind = np.unique(tokens, return_inverse=True, axis=0)
        return (
            batch_counts,
            counts.reshape((-1, 1)),
            tokenize_asv(unique_t),
            unique_ind[indices],
            y_output,
            encoder_out,
            ob_ids,
            s_ids,
        )


if __name__ == "__main__":
    import numpy as np

    ug = MultiDepthGenerator(
        table="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom",
        tree_path="/home/kalen/aam-research-exam/research-exam/agp/data/agp-aligned.nwk",
        metadata="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-healthy.txt",
        metadata_column="host_age",
        sample_depths=[100, 1000],
        # shift=0.0,
        scale="minmax",
        gen_new_tables=True,
        max_token_per_sample=2048,
        batch_size=4,
    )
    for x, y in ug:
        print(x, y)
        break
    # # model = tf.keras.models.load_model(
    # #     "/home/kalen/aam-research-exam/research-exam/healty-age-regression/unifrac-regressor-LAMB-norm/model.keras",
    # #     compile=False,
    # # )
    # print(data_obj)
    # model = tf.keras.models.load_model(
    #     "/home/kalen/aam-research-exam/research-exam/healty-age-regression/profile-unifrac-regressor/model.keras", compile=False
    # )
    # for x, y in data_obj["dataset"].take(1):
    #     y_target, encoder_target = y
    #     batch_counts, tokens, indicies, counts = x
    #     group_dim = tf.shape(encoder_target)[-1]
    #     batch_counts = tf.reshape(batch_counts, shape=[-1, group_dim])
    #     batch_sums = tf.pad(tf.reduce_sum(batch_counts[:-1], axis=-1, keepdims=True), [[1, 0], [0, 0]])
    #     batch_sums = tf.squeeze(batch_sums, axis=-1)
    #     batch_sums = tf.math.cumsum(batch_sums, axis=0)
    #     # print(batch_counts, batch_sums, tf.reduce_sum(batch_counts, axis=-1), tf.reduce_sum(batch_counts), indicies.shape)

    #     def _process_batch(inputs):
    #         bi_batch_counts, prev_batch_sums = inputs
    #         bi_total = tf.reduce_sum(bi_batch_counts)
    #         bi_indices = indicies[prev_batch_sums : prev_batch_sums + bi_total]
    #         bi_counts = counts[prev_batch_sums : prev_batch_sums + bi_total]
    #         print("WHAT???", bi_total, bi_indices.shape)
    #         return model((bi_batch_counts, tokens, bi_indices, bi_counts), training=True)

    #     output = tf.map_fn(
    #         _process_batch,
    #         (batch_counts, batch_sums),
    #         fn_output_signature=(
    #             tf.TensorSpec(shape=[None, 128], dtype=tf.float32),
    #             tf.TensorSpec(shape=[None, 128], dtype=tf.float32),
    #         ),
    #     )
