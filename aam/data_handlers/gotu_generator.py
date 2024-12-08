from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf
from biom import Table
from bp import parse_newick
from skbio import DistanceMatrix

from aam.data_handlers.unifrac_generator import UniFracGenerator


class GOTUGenerator(UniFracGenerator):
    def __init__(self, **kwargs):
        super(GOTUGenerator, self).__init__(**kwargs)
        self.token_mapping = self.map_tokens()

    def _create_encoder_target(self, table: Table) -> DistanceMatrix:
        return super(GOTUGenerator, self)._create_encoder_target(table)

    def _encoder_output(
        self,
        encoder_target: DistanceMatrix,
        sample_ids: Iterable[str],
        ob_ids: list[str],
    ) -> np.ndarray[float]:
        return super(GOTUGenerator, self)._encoder_output(
            encoder_target, sample_ids, ob_ids
        )

    def map_tokens(self) -> dict:
        """
        Encodes GOTUs based on a post-order traversal of a Newick tree and tokenizes
        the observation IDs of the BIOM table.

        Args:
            bp_tree_fp (str): File path to the Newick tree.

        Returns:
            dict: A dictionary mapping observation IDs to their token values.
        """
        print("Loading Tree")
        bp_tree = parse_newick(open(self.tree_path).read())
        print("Tree Loaded, generating token ID's")
        name_list = []
        token_mapping = {}
        token_id = 0
        for i in range(1, ((bp_tree.__len__()) * 2)):
            if not bp_tree.isleaf(i):
                continue
            if bp_tree.name(i) is not None:
                node_name = bp_tree.name(i)
                if len(node_name) == 10:
                    name_list.append(node_name)
                    token_mapping[node_name] = token_id
                    token_id += 1

        return token_mapping

    def gotu_tokens(self, indicies: np.ndarray) -> np.ndarray:
        obs_ids = self.obs_ids[indicies]
        return np.array([self.token_mapping[id] for id in obs_ids]).reshape((-1, 1))


# if __name__ == "__main__":
#     gotu_gen = GOTUGenerator(
#         table="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/gotu_ordered_table_filtered.biom",
#         tree_path="/home/jokirkland/data/trees/2022.10.phylogeny.asv.nwk",
#         metadata="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/metag_metadata.tsv",
#         metadata_column="Age",
#         shift=0.0,
#         scale=100.0,
#         gen_new_tables=True,
#         is_16S=False,
#     )
#     print("Finished creating generator")
#     gotu_data = gotu_gen.get_data()
#     gotu_dataset = gotu_data["dataset"]

#     ug = UniFracGenerator(
#         table="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/asv_ordered_table.biom",
#         tree_path="/home/jokirkland/data/trees/2022.10.phylogeny.asv.nwk",
#         metadata="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/metag_metadata.tsv",
#         metadata_column="Age",
#         shift=0.0,
#         scale=100.0,
#         gen_new_tables=True,
#     )
#     print("Finished creating 16S data")
#     asv_data = ug.get_data()
#     asv_dataset = asv_data["dataset"]
#     for i, (x, y) in enumerate(asv_dataset):
#         print("ASV_DATSET_VALUES")
#         print(f"Printing X Data: {x}\nPrinting Y Data {y}")
#         break

#     for i, (x, y) in enumerate(gotu_dataset):
#         print("GOTU_DATSET_VALUES")
#         print(f"Printing X Data: {x}\nPrinting Y Data {y}")
#         break
#     full_dataset = tf.data.Dataset.zip((asv_dataset, gotu_dataset))

#     for i, (x, y) in enumerate(full_dataset):
#         print("FULL MERGED DATASET")
#         print(f"Printing X Data: {x}\nPrinting Y Data {y}")
#         break
