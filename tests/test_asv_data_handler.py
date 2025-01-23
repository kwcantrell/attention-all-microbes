import numpy as np
import pytest
from skbio import TreeNode

from aam.data_handlers.asv_generator import ASVGenerator


def test_node_order():
    with open("reference-tree.nwk", "r") as f:
        reference_tree: TreeNode = TreeNode.read(f)
    ag = ASVGenerator(tree="reference-tree.nwk", shuffle=False, return_asv_ids=False, drop_remainder=False)

    reference_names = []
    for node in reference_tree.preorder(include_self=True):
        if not node.is_root():
            node.length += node.parent.length

        if node.is_tip():
            reference_names.append(node.name.lower())

    generator_names = []
    gen_tokens = []
    for i in range(len(ag)):
        tokens, names = ag[i]
        for asv_tokens in list(tokens):
            gen_tokens.append(asv_tokens)

        for name in list(names):
            generator_names.append(name)

    assert len(generator_names) == len(reference_names)
    for ref_name, gen_name in zip(reference_names, generator_names):
        assert ref_name == gen_name


def test_tip_tip_distance():
    with open("reference-tree.nwk", "r") as f:
        reference_tree: TreeNode = TreeNode.read(f)
    ag = ASVGenerator(
        tree="reference-tree.nwk",
        shuffle=False,
        sequence_batch_size=512,
        pairwise_batch_size=512,
        return_asv_ids=False,
        drop_remainder=False,
    )
    for node in reference_tree.preorder(include_self=True):
        if node.length is None:
            node.length = 0

        if node.name is not None:
            node.name = node.name.lower()

    lookup = [
        "",
        "a",
        "c",
        "g",
        "t",
    ]

    def tokens_to_asv(tokens):
        asv = ""
        for token in list(tokens):
            asv += lookup[token]
        return asv

    def _distance_matrix(asvs):
        num_asvs = len(asvs)
        distances = np.zeros((num_asvs, num_asvs), dtype=np.float32)
        for i in range(len(asvs)):
            for j in range(i + 1, num_asvs, 1):
                i_node = reference_tree.find(asvs[i])
                j_node = reference_tree.find(asvs[j])
                distances[i, j] = i_node.distance(j_node) / ag.max_dist_to_root
        distances = distances
        return distances + distances.T

    for i in range(len(ag)):
        tokens, tip_distances = ag[i]
        asvs = []
        for asv_tokens in list(tokens):
            asvs.append(tokens_to_asv(asv_tokens))
        reference_distances = _distance_matrix(asvs)
        assert np.allclose(reference_distances, tip_distances, atol=1e-7)


if __name__ == "__main__":
    test_tip_tip_distance()
