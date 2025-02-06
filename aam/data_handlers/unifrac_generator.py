from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import skbio.diversity as diversity
import tensorflow as tf
from biom import Table
from biom.util import biom_open
from skbio import DistanceMatrix
from unifrac import unweighted

from aam.data_handlers.generator_dataset import GeneratorDataset


class UniFracGenerator(GeneratorDataset):
    def __init__(self, unifrac_metric="unifrac", **kwargs):
        super(UniFracGenerator, self).__init__(**kwargs)
        self.encoder_dtype = np.float32

    def _create_encoder_target(self) -> DistanceMatrix:
        super(UniFracGenerator, self)._create_encoder_target()
        print("creating unifrac targets...")

        # NOTE: unweighted returns an "approximation" and may
        # introduce a bit of noise although it should be highly
        # negligible
        return unweighted(self.rarefied_table, self.tree)

    def _encoder_output(self, sample_ids: Iterable[str]) -> np.ndarray[float]:
        return self.encoder_target.filter(sample_ids).data


def get_dataset(gen: UniFracGenerator):
    enqueuer = tf.keras.utils.OrderedEnqueuer(gen, use_multiprocessing=True)
    enqueuer.start(workers=2, max_queue_size=gen.steps_per_epoch)
    gen.stop = lambda: enqueuer.stop(0.1)

    if not gen.return_sample_ids:
        y_type = (
            tf.TensorSpec(shape=[gen.batch_size, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[gen.batch_size, gen.batch_size], dtype=tf.float32),
        )
    else:
        y_type = tf.TensorSpec(shape=(gen.batch_size), dtype=tf.string)

    dataset = tf.data.Dataset.from_generator(
        enqueuer.get,
        output_signature=(
            (
                tf.TensorSpec(shape=[None, 150], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            ),
            y_type,
        ),
    )

    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    import numpy as np

    ug = UniFracGenerator(
        table="/home/jokirkland/data/aam/sanity/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom",
        tree_path="/home/jokirkland/data/aam/sanity/agp-aligned.nwk",
        metadata="/home/jokirkland/data/aam/sanity/agp-healthy.txt",
        metadata_column="host_age",
        shift=0.0,
        scale=100.0,
        gen_new_tables=True,
        return_sample_ids=False,
    )
    print(ug[0])
    # dataset = get_dataset(ug)
    # for x, y in dataset.take(1):
    #     print(x, y)
    # ug.stop()
