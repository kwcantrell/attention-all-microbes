from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table
from sklearn import preprocessing

from aam.data_handlers.generator_dataset import GeneratorDataset, add_lock


class TaxonomyGenerator(GeneratorDataset):
    taxon_field = "Taxon"
    levels = [f"Level {i}" for i in range(1, 8)]
    taxonomy_fn = "taxonomy.tsv"

    def __init__(self, taxonomy, tax_level: int, **kwargs):
        super().__init__(**kwargs)
        self.tax_level = f"Level {tax_level}"
        self.taxonomy = taxonomy

        self.encoder_target = self._create_encoder_target(self.rarefy_table)
        self.encoder_dtype = np.int32
        self.encoder_output_type = tf.TensorSpec(
            shape=[self.batch_size, None], dtype=tf.int32
        )

    def _create_encoder_target(self, table: Table) -> None:
        if not hasattr(self, "_taxonomy"):
            return None
        obs = table.ids(axis="observation")
        return self.taxonomy.loc[obs, "token"]

    def _encoder_output(
        self, encoder_target: pd.Series, sample_ids: Iterable[str], obs_ids: list[str]
    ):
        tax_tokens = [encoder_target.loc[obs] for obs in obs_ids]
        max_len = max([len(tokens) for tokens in tax_tokens])
        return np.array([np.pad(t, [[0, max_len - len(t)]]) for t in tax_tokens])

    @property
    def taxonomy(self) -> pd.DataFrame:
        return self._taxonomy

    @taxonomy.setter
    @add_lock
    def taxonomy(self, taxonomy: Union[str, pd.DataFrame]):
        if hasattr(self, "_taxon_set"):
            raise Exception("Taxon already set")
        if taxonomy is None:
            self._taxonomy = taxonomy
            return

        if isinstance(taxonomy, str):
            taxonomy = pd.read_csv(taxonomy, sep="\t", index_col=0)
            if self.taxon_field not in taxonomy.columns:
                raise Exception("Invalid taxonomy: missing 'Taxon' field")

        taxonomy[self.levels] = taxonomy[self.taxon_field].str.split("; ", expand=True)
        taxonomy = taxonomy.loc[self._table.ids(axis="observation")]

        if self.tax_level not in self.levels:
            raise Exception(f"Invalid level: {self.tax_level}")

        level_index = self.levels.index(self.tax_level)
        levels = self.levels[: level_index + 1]
        taxonomy = taxonomy.loc[:, levels]
        taxonomy.loc[:, "class"] = taxonomy.loc[:, levels].agg("; ".join, axis=1)

        le = preprocessing.LabelEncoder()
        taxonomy.loc[:, "token"] = le.fit_transform(taxonomy["class"])
        taxonomy.loc[:, "token"] += 1  # shifts tokens to be between 1 and n
        print(
            "min token:", min(taxonomy["token"]), "max token:", max(taxonomy["token"])
        )
        self.num_tokens = (
            max(taxonomy["token"]) + 1
        )  # still need to add 1 to account for shift
        self._taxonomy = taxonomy


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
    sd = TaxonomyGenerator(
        tax_level=7,
        table=table,
        # metadata=metadata,
        # metadata_column="host_age",
        taxonomy="/home/kalen/aam-research-exam/research-exam/healty-age-regression/taxonomy.tsv",
    )
    data = sd.get_data()
    for data in data["dataset"]:
        inputs, y = data
        print(y)
        break
