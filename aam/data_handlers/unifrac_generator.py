from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import tensorflow as tf
from biom import Table
from biom.util import biom_open
from skbio import DistanceMatrix
from unifrac import faith_pd, unweighted

from aam.data_handlers.asv_generator import TOKENIZER
from aam.data_handlers.generator_dataset import GeneratorDataset


class UniFracGenerator(GeneratorDataset):
    def __init__(self, tree_path: str, unifrac_metric="unifrac", **kwargs):
        super(UniFracGenerator, self).__init__(**kwargs)
        self.tree_path = tree_path
        self.unifrac_metric = unifrac_metric

        self.encoder_target = self._create_encoder_target(self.rarefy_table)
        self.encoder_dtype = np.float32

    def _create_encoder_target(self, table: Table) -> DistanceMatrix:
        if not hasattr(self, "tree_path"):
            return None

        random = np.random.random(1)[0]
        temp_path = f"/tmp/temp{random}.biom"
        with biom_open(temp_path, "w") as f:
            table.to_hdf5(f, "aam")
        if self.unifrac_metric == "unifrac":
            distances = unweighted(temp_path, self.tree_path)
        else:
            distances = faith_pd(temp_path, self.tree_path)
        os.remove(temp_path)
        return distances

    def _encoder_output(
        self,
        encoder_target: DistanceMatrix,
        sample_ids: Iterable[str],
        ob_ids: list[str],
    ) -> np.ndarray[float]:
        if self.unifrac_metric == "unifrac":
            return encoder_target.filter(sample_ids).data
        else:
            return encoder_target.loc[sample_ids].to_numpy().reshape((-1, 1))


def get_dataset(gen: UniFracGenerator):
    enqueuer = tf.keras.utils.OrderedEnqueuer(gen, use_multiprocessing=True)
    enqueuer.start(workers=4, max_queue_size=128)

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
                tf.TensorSpec(shape=[gen.batch_size], dtype=tf.int32),
                tf.TensorSpec(shape=[None], dtype=tf.string),
                tf.TensorSpec(shape=[None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            ),
            y_type,
        ),
    )

    def tokenize_asv(inputs, targets):
        batch_counts, asvs, inx, counts = inputs
        tokens = TOKENIZER(asvs)
        mask = tokens > 0
        tokens = tf.cast(tokens, dtype=tf.int32) - tf.cast(mask, dtype=tf.int32)
        return (batch_counts, tokens, inx, counts), targets

    dataset = dataset.map(tokenize_asv, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    import numpy as np

    ug = UniFracGenerator(
        table="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom",
        tree_path="/home/kalen/aam-research-exam/research-exam/agp/data/agp-aligned.nwk",
        metadata="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-healthy.txt",
        metadata_column="host_age",
        shift=0.0,
        scale=100.0,
        gen_new_tables=True,
        return_sample_ids=True,
    )
    dataset = get_dataset(ug)
    for x, y in dataset.take(1):
        print(y)
    # data = ug.get_data_by_id(ug.rarefy_tables.ids()[:16])
    # for x, y in data["dataset"]:
    #     print(y)
    #     break
