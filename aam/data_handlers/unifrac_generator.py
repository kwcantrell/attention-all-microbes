from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import tensorflow as tf
from biom import Table
from biom.util import biom_open
from skbio import DistanceMatrix
from unifrac import unweighted

from aam.data_handlers.generator_dataset import GeneratorDataset


class UniFracGenerator(GeneratorDataset):
    def __init__(self, tree_path: str, **kwargs):
        super().__init__(**kwargs)
        self.tree_path = tree_path
        if self.batch_size % 2 != 0:
            raise Exception("Batch size must be multiple of 2")
        self.encoder_target = self._create_encoder_target(self.rarefy_table)
        self.encoder_dtype = np.float32
        self.encoder_output_type = tf.TensorSpec(
            shape=[self.batch_size, self.batch_size], dtype=tf.float32
        )

    def _create_encoder_target(self, table: Table) -> DistanceMatrix:
        if not hasattr(self, "tree_path"):
            return None

        random = np.random.random(1)[0]
        temp_path = f"/tmp/temp{random}.biom"
        with biom_open(temp_path, "w") as f:
            table.to_hdf5(f, "aam")
        distances = unweighted(temp_path, self.tree_path)
        os.remove(temp_path)
        return distances

    def _encoder_output(
        self,
        encoder_target: DistanceMatrix,
        sample_ids: Iterable[str],
        ob_ids: list[str],
    ) -> np.ndarray[float]:
        return encoder_target.filter(sample_ids).data


if __name__ == "__main__":
    from aam.data_handlers import UniFracGenerator

    ug = UniFracGenerator(
        table="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom",
        tree_path="/home/kalen/aam-research-exam/research-exam/agp/data/agp-aligned.nwk",
        metadata="/home/kalen/aam-research-exam/research-exam/healty-age-regression/agp-healthy.txt",
        metadata_column="host_age",
        shift=0.0,
        scale=100.0,
        gen_new_tables=True,
    )
    data = ug.get_data()
    for i, (x, y) in enumerate(data["dataset"]):
        print(y)
        break

    # data = ug.get_data_by_id(ug.rarefy_tables.ids()[:16])
    # for x, y in data["dataset"]:
    #     print(y)
    #     break
