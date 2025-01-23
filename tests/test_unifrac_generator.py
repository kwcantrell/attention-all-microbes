import numpy as np
import tensorflow as tf
from biom import load_table
from biom.util import biom_open
from bp import parse_newick, to_skbio_treenode
from skbio import TreeNode
from unifrac import unweighted

from aam.data_handlers.unifrac_generator import UniFracGenerator


def test_unweighted():
    table_path = "agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom"
    tree_path = "agp-aligned.nwk"
    metadata_path = "agp-healthy.txt"
    batch_size = 8
    gd = UniFracGenerator(
        table=table_path,
        tree_path=tree_path,
        metadata=metadata_path,
        metadata_column="host_age",
        scale="minmax",
        gen_new_tables=True,
        batch_size=batch_size,
        shuffle=False,
    )
    reference_tree: TreeNode = to_skbio_treenode(parse_newick(open(tree_path).read()))
    for node in reference_tree.preorder(include_self=True):
        if node.is_root():
            node.root_to_node = []
        else:
            node.root_to_node = [a for a in node.parent.root_to_node]
            node.root_to_node.append(node.parent)

    print("load table")
    table = load_table(table_path).subsample(5000)
    print("done")
    sample_ids = table.ids()
    o_ids = table.ids(axis="observation")

    temp_path = "unifrac-table.biom"
    with biom_open(temp_path, "w") as f:
        table.to_hdf5(f, "aam")

    print("unifrac...")
    distances = gd.encoder_target
    print("done")

    def _extract_sample(samples):
        table_data = [table.data(s, dense=False) for s in samples]
        unique_obs = set()
        sample_indices = []
        for data in table_data:
            obs_indices, _ = data.tocoo().coords
            unique_obs = unique_obs.union(obs_indices)
            sample_indices.append(obs_indices)

        tips = {o: reference_tree.find(o_ids[o]) for o in unique_obs}

        distances = np.zeros((len(samples), len(samples)))
        for i in range(len(sample_indices)-1):
            s_i = {tips[o] for o in sample_indices[i]}
            for j in range(i+1, len(sample_indices), 1):
                s_j = {tips[o] for o in sample_indices[j]}
        
                share = s_i.intersection(s_j)
                share_length = 
                diff = s_i.symmetric_difference(s_j)


    for i in range(len(gd)):
        print(f"testing batch {i}")
        batch_ids = sample_ids[i * batch_size : (i + 1) * batch_size]
        print(_extract_sample(batch_ids))
        break
        # (tokens, indices, asv_indices, counts), y = gd[i]
        # batch_tokens, batch_counts = batch_embeddings(tokens, asv_indices, indices, counts)
        # batch_gen = []
        # for sample, counts in zip(list(batch_tokens.numpy()), list(batch_counts.numpy())):
        #     sample_asvs = []
        #     sample_mask = counts.reshape((-1)) > 0
        #     sample = sample[sample_mask]

        #     for asv_tokens in sample:
        #         sample_asvs.append(tokens_to_asv(asv_tokens))
        #     batch_gen.append((np.hstack(sample_asvs), counts[sample_mask]))

        # for table_sample, gen_sample in zip(table_batch, batch_gen):
        #     table_obs, table_counts = table_sample
        #     gen_obs, gen_counts = gen_sample

        #     assert np.array_equal(table_counts.reshape((-1)), gen_counts.reshape((-1)))
        #     assert np.array_equal(table_obs, gen_obs)


# def test_batch_reconstruction():


if __name__ == "__main__":
    test_unweighted()
