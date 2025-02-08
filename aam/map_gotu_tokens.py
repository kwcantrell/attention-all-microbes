import json

import biom
from biom import table
from bp import parse_newick, to_skbio_treenode


def save_json(output_dict: dict, output_dir: str):
    with open(output_dir, "w") as fp:
        json.dump(output_dict, fp)


def create_tree_dict(nwk_tree_fp: str) -> dict:
    bp_tree = to_skbio_treenode(parse_newick(open(nwk_tree_fp).read()))
    name_to_rank = {}
    rank_to_name = {}
    i = 0
    for node in bp_tree.postorder():
        if (
            node.name is not None
            and len(node.name) <= 25
            and "__" not in node.name
            and "G" in node.name[0]
        ):
            name_to_rank[node.name] = i
            rank_to_name[i] = node.name
            i += 1

    return name_to_rank, rank_to_name


if __name__ == "__main__":
    print("Starting traversal....")
    name2rank, rank2name = create_tree_dict(
        "/home/jokirkland/data/trees/2022.10.phylogeny.asv.nwk"
    )
    save_json(name2rank, "/home/jokirkland/data/trees/gotu_node_dict.json")
    save_json(rank2name, "/home/jokirkland/data/trees/gotu_rank_dict.json")

    print("dict saved...")
