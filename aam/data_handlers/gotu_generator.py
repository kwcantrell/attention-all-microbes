from __future__ import annotations

from bp import parse_newick
from biom import load_table, Table
from aam.data_handlers.unifrac_generator import UniFracGenerator

class GOTUGenerator(UniFracGenerator):
    def __init__(self, gotu_table_fp: str, **kwargs):
        super().__init__(**kwargs)
        self.gotu_table_fp = gotu_table_fp
        
    def parse_gotu_table(self):
        pass
        
           
    def gotu_encoder(self) -> dict:
        """
        Encodes GOTUs based on a post-order traversal of a Newick tree and tokenizes
        the observation IDs of the BIOM table.

        Args:
            bp_tree_fp (str): File path to the Newick tree.

        Returns:
            dict: A dictionary mapping observation IDs to their token values.
        """
        bp_tree = parse_newick(open(self.tree_path).read())
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

    
if __name__ == "__main__":
