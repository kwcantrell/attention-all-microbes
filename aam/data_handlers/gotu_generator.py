from __future__ import annotations

import numpy as np
from bp import parse_newick

from aam.data_handlers.unifrac_generator import UniFracGenerator


class GOTUGenerator(UniFracGenerator):
    def __init__(self, **kwargs):
        super(GOTUGenerator, self).__init__(**kwargs)
        self.token_mapping = self.map_tokens()

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
#     data = gotu_gen.get_data()
#     for i, (x, y) in enumerate(data["dataset"]):
#         print(y[1], np.log1p(y[1]), np.sqrt(y[1]))
#         break
