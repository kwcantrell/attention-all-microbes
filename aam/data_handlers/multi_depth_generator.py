from __future__ import annotations

import os
from functools import wraps
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table

from aam.data_handlers import GeneratorDataset


class MultiDepthGenerator(GeneratorDataset):
    def __init__(
        self,
        table: Union[str, Table],
        sample_depths: List[int],
        tree_path=None,
        unifrac_metric="unifrac",
        gen_new_table_frequency=3,
        **kwargs,
    ):
        if isinstance(table, str):
            table = load_table(table)
        from aam.data_handlers import UniFracGenerator

        super(MultiDepthGenerator, self).__init__(**kwargs)
        max_depth = max(sample_depths)
        self.sample_depths = sample_depths
        sample_mask = table.sum(axis="sample") >= max_depth
        keep_samples = table.ids()[sample_mask]
        table = table.filter(keep_samples)
        table = table.remove_empty()
        self.table = table

        kwargs["tree_path"] = tree_path
        kwargs["unifrac_metric"] = unifrac_metric
        self.generators = [UniFracGenerator(table=self.table, rarefy_depth=depth, **kwargs) for depth in sample_depths]

        self.sample_mask = np.logical_and.reduce([gen.sample_mask for gen in self.generators])
        self.sample_indices = np.arange(len(self.table.ids()))[self.sample_mask]
        self.size = len(self.table.ids())
        self.steps_per_epoch = (self.size // self.batch_size) * self.repeat
        self.encoder_output_type = tf.TensorSpec(
            shape=[len(sample_depths) * self.batch_size, self.batch_size], dtype=tf.float32
        )
        self.gen_new_table_frequency = gen_new_table_frequency

    def _sample_data(
        self, samples: np.ndarray, table_data=None, y_data=None, encoder_target=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = [
            gen._sample_data(samples, td, yd, et)
            for gen, td, yd, et in zip(self.generators, table_data, y_data, encoder_target)
        ]

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
            shift += np.sum(bc)

        batch_counts = np.concatenate(batch_counts)
        counts = np.concatenate(counts)
        tokens = np.concatenate(tokens)
        indices = np.concatenate(indices)
        y_output = np.concatenate(y_output)
        encoder_out = np.concatenate(encoder_out)
        ob_ids = np.concatenate(ob_ids)
        s_ids = np.concatenate(s_ids)

        return batch_counts, counts.reshape((-1, 1)), tokens, indices, y_output, encoder_out, ob_ids, s_ids

    def _epoch_samples(self, epoch, table_data, y_data, encoder_target, sample_mask, sample_indices):
        if self.gen_new_tables and epoch > 0 and epoch % self.gen_new_table_frequency == 0:
            new_table_data, new_y_data, new_encoder_target, sample_masks = [], [], [], []
            for gen, _td, _yd, _et in zip(self.generators, table_data, y_data, encoder_target):
                td, yd, et, sm, _ = gen._epoch_samples(epoch, _td, _yd, _et, sample_mask, sample_indices)
                new_table_data.append(td)
                new_y_data.append(yd)
                new_encoder_target.append(et)
                sample_masks.append(sm)

            sample_mask = np.logical_and.reduce(sample_masks)
            sample_indices = np.arange(len(self.table.ids()))[sample_mask]
            table_data = new_table_data
            y_data = new_y_data
            encoder_target = new_encoder_target

        if self.shuffle:
            print("shuffling...")
            np.random.shuffle(sample_indices)

        return table_data, y_data, encoder_target, sample_mask, sample_indices

    def _create_epoch_generator(self, include_seq_id, include_sample_ids):
        def generator():
            processed = 0
            table_data = [gen.table_data for gen in self.generators]
            y_data = [gen.y_data for gen in self.generators]
            encoder_target = [gen.encoder_target for gen in self.generators]
            sample_mask = np.logical_and.reduce([gen.sample_mask for gen in self.generators])
            sample_indices = self.sample_indices
            for epoch in range(self.epochs):
                print(f"Finished epcoh: {epoch} processed {processed}")
                processed = 0
                minibatch = 0
                table_data, y_data, encoder_target, sample_mask, sample_indices = self._epoch_samples(
                    epoch, table_data, y_data, encoder_target, sample_mask, sample_indices
                )

                def sample_data(minibatch):
                    samples = self._minibatch_indices(minibatch, sample_indices)
                    return self._sample_data(samples, table_data, y_data, encoder_target)

                while not self._epoch_complete(processed):
                    (batch_counts, counts, tokens, indices, y_output, encoder_out, ob_ids, s_ids) = sample_data(minibatch)

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
                                    [o.astype(t) for o, t in zip(encoder_out, self.generators[0].encoder_dtype)]
                                )
                            else:
                                encoder_out = encoder_out.astype(self.generators[0].encoder_dtype)

                            if output is not None:
                                output = (output, encoder_out)
                            else:
                                output = encoder_out

                        if include_seq_id:
                            output = (*output, ob_ids)
                        if include_sample_ids:
                            output = (*output, s_ids)

                        if output is not None:
                            yield (table_output, output)
                        else:
                            yield table_output
                    minibatch += 1

        return generator

    def get_data(self, include_seq_id=False, include_sample_ids=False):
        generator = self._create_epoch_generator(include_seq_id, include_sample_ids)

        if self.is_16S:
            output_sig = (
                tf.TensorSpec(shape=[len(self.sample_depths) * self.batch_size], dtype=tf.int32),
                tf.TensorSpec(shape=[None, self.max_bp], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            )
        else:
            output_sig = (
                tf.TensorSpec(shape=[len(self.sample_depths) * self.batch_size, None, 1], dtype=tf.int32),
                tf.TensorSpec(shape=[len(self.sample_depths) * self.batch_size, None, 1], dtype=tf.int32),
            )

        y_output_sig = None
        y_output_sig = tf.TensorSpec(shape=[len(self.sample_depths) * self.batch_size, 1], dtype=tf.float32)
        y_output_sig = (y_output_sig, self.encoder_output_type)
        output_sig = (output_sig, y_output_sig)

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
        }
        return data_obj


if __name__ == "__main__":
    import numpy as np

    from aam.data_handlers import UniFracGenerator

    ug = MultiDepthGenerator(
        table="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom",
        tree_path="/home/kalen/aam-research-exam/research-exam/agp/data/agp-aligned.nwk",
        metadata="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-healthy.txt",
        metadata_column="host_age",
        sample_depths=[100, 1000, 5000],
        shift=0.0,
        scale=100.0,
        gen_new_tables=True,
        max_token_per_sample=100,
        batch_size=4,
    )
    data_obj = ug.get_data()
    model = tf.keras.models.load_model(
        "/home/kalen/aam-research-exam/research-exam/healty-age-regression/unifrac-regressor-LAMB-norm/model.keras",
        compile=False,
    )
    for x, y in data_obj["dataset"].take(1):
        y_target, encoder_target = y
    print(encoder_target)
    shape = tf.shape(encoder_target)
    batch_dim = shape[0]
    group_dim = shape[-1]
    groups = batch_dim // group_dim
    print(tf.reshape(encoder_target, shape=[groups, group_dim, group_dim]))
    # data = ug.get_data()
    # for i, (x, y) in enumerate(data["dataset"]):
    #     print(y[1], np.log1p(y[1]), np.sqrt(y[1]))
    #     break

    # data = ug.get_data_by_id(ug.rarefy_tables.ids()[:16])
    # for x, y in data["dataset"]:
    #     print(y)
    #     break
