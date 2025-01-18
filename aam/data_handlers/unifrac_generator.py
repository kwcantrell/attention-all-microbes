from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import tensorflow as tf
from biom import Table
from biom.util import biom_open
from skbio import DistanceMatrix
from unifrac import faith_pd, unweighted

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


if __name__ == "__main__":
    import numpy as np

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

    for i, (x, y) in enumerate(ug):
        print(y[1], np.log1p(y[1]), np.sqrt(y[1]))
        break

    # data = ug.get_data_by_id(ug.rarefy_tables.ids()[:16])
    # for x, y in data["dataset"]:
    #     print(y)
    #     break
