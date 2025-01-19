from __future__ import annotations

from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table

from aam.data_handlers.asv_generator import TOKENIZER
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
        return_sample_ids=False,
        epochs=1000,
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
        print(self.size, self.steps_per_epoch, [g.steps_per_epoch for g in self.generators])
        self.shuffle = shuffle
        self.gen_new_table_frequency = gen_new_table_frequency
        self.epochs_since_last_table = 0
        self.return_sample_ids = return_sample_ids
        self.epochs = epochs
        self.update_sample_indices()
        self.on_epoch_end()

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size

        sample_indices = self.sample_indices[start:end]
        sample_ids = self.common_ids[sample_indices]

        def _samples(i, sample_ids, gen):
            _, _sample_indices, _ = np.intersect1d(gen.rarefy_table.ids(), sample_ids, return_indices=True, assume_unique=True)
            return gen._sample_data(_sample_indices)

        outputs = [_samples(i, sample_ids, gen) for i, gen in enumerate(self.generators)]
        combined_outputs = self._sample_data(outputs)

        (batch_counts, counts, tokens, indices, y_output, encoder_out, ob_ids, s_ids) = combined_outputs

        lookup = {
            "a": 1,
            "c": 2,
            "g": 3,
            "t": 3,
        }

        def map(asv):
            asv = asv.lower()
            return [lookup[c] for c in asv]

        tokens = [map(o) for o in tokens]
        if counts is not None:
            table_output = (
                batch_counts.astype(np.int32),
                tokens,
                indices.astype(np.int32),
                counts.astype(np.int32),
            )

            if self.return_sample_ids:
                return (table_output, s_ids)

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
        self.common_ids = np.intersect1d(self.generators[0].sample_ids, self.generators[1].sample_ids, assume_unique=True)
        self.sample_indices = np.arange(len(self.common_ids))

        fill_out = (self.size // len(self.sample_indices)) + 1

        if self.shuffle:
            np.random.shuffle(self.sample_indices)

        if fill_out > 0:
            self.sample_indices = np.repeat([self.sample_indices], repeats=fill_out, axis=0).reshape((-1))

    def on_epoch_end(self):
        if self.epochs_since_last_table >= self.gen_new_table_frequency and self.shuffle:
            print("creating new data...")
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
            unique_t,
            unique_ind[indices],
            y_output,
            encoder_out,
            ob_ids,
            s_ids,
        )


def get_dataset(gen: MultiDepthGenerator):
    enqueuer = tf.keras.utils.OrderedEnqueuer(gen, use_multiprocessing=True)
    enqueuer.start(workers=2, max_queue_size=gen.steps_per_epoch)
    gen.stop = enqueuer.stop

    if not gen.return_sample_ids:
        y_type = tf.TensorSpec(shape=[gen.batch_size * len(gen.generators), gen.batch_size], dtype=tf.float32)
    else:
        y_type = tf.TensorSpec(shape=(gen.batch_size * len(gen.generators)), dtype=tf.string)

    dataset = tf.data.Dataset.from_generator(
        enqueuer.get,
        output_signature=(
            (
                tf.TensorSpec(shape=[gen.batch_size * len(gen.generators)], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 150], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            ),
            (tf.TensorSpec(shape=[gen.batch_size * len(gen.generators), 1], dtype=tf.float32), y_type),
        ),
    )

    # def tokenize_asv(inputs, targets):
    #     batch_counts, asvs, inx, counts = inputs
    #     tokens = TOKENIZER(asvs)
    #     mask = tokens > 0
    #     tokens = tf.cast(tokens, dtype=tf.int32) - tf.cast(mask, dtype=tf.int32)
    #     return (batch_counts, tokens, inx, counts), targets

    # dataset = dataset.map(tokenize_asv)
    # dataset = dataset.map(tokenize_asv, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    dataset = dataset.prefetch(10)
    return dataset


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

    dataset = get_dataset(ug)
    for x in dataset.take(1):
        print(x)
