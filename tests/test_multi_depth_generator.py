import numpy as np
import tensorflow as tf
from biom import load_table

from aam.data_handlers.multi_depth_generator import MultiDepthGenerator

lookup = [
    "",
    "A",
    "C",
    "G",
    "T",
]


def tokens_to_asv(tokens):
    asv = ""
    for token in list(tokens):
        asv += lookup[token]
    return asv


def batch_embeddings(asv_embeddings, batch_indicies, asv_indices, counts):
    emb_dim = tf.shape(asv_embeddings)[-1]
    if asv_indices is not None:
        asv_embeddings = tf.gather(asv_embeddings, asv_indices)
    batch_shape = tf.reduce_max(batch_indicies[:, 0]) + 1
    max_unique = tf.reduce_max(batch_indicies[:, 1]) + 1
    batch_embeddings = tf.scatter_nd(batch_indicies, asv_embeddings, shape=[batch_shape, max_unique, emb_dim])
    counts = tf.scatter_nd(batch_indicies, counts, shape=[batch_shape, max_unique, 1])
    return batch_embeddings, counts


def test_sample_info():
    table_path = "/home/kalen/amplicon-gpt/tests/agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom"
    tree_path = "/home/kalen/aam-research-exam/research-exam/agp/data/agp-aligned.nwk"
    metadata_path = "agp-healthy.txt"
    batch_size = 8
    gd = MultiDepthGenerator(
        table=table_path,
        tree_path=tree_path,
        metadata=metadata_path,
        metadata_column="host_age",
        scale="minmax",
        gen_new_tables=True,
        batch_size=batch_size,
        shuffle=False,
        sample_depths=[1000, 5000],
    )

    tables = [g.rarefied_table for g in gd.generators]

    common_ids = set()
    for table in tables:
        common_ids.update(table.ids())

    assert common_ids == set(gd.common_ids)

    inputs, outputs = gd[0]

    # check unifrac is correct
    batch_ids = gd.common_ids[: gd.batch_size]
    distances = [gen._encoder_output(batch_ids) for gen in gd.generators]
    distancecs = np.vstack(distances)

    assert np.array_equal(distancecs, outputs[1])

    gen_tokens, gen_counts = batch_embeddings(*inputs)

    sample_asvs, sample_counts = [], []
    for gen in gd.generators:
        obs_ids = gen.rarefied_table.ids(axis="observation")
        for s_id in batch_ids:
            sample_data = gen.rarefied_table.data(s_id, dense=False).tocoo()
            (obs_idx, _), counts = sample_data.coords, sample_data.data

            sorted_indices = np.argsort(counts)
            sorted_indices = sorted_indices[::-1]

            sample_asvs.append(obs_ids[obs_idx[sorted_indices]])
            sample_counts.append(counts[sorted_indices, np.newaxis])

    lookup = ["", "A", "C", "G", "T"]

    def map(asv_tokens):
        return "".join([lookup[t] for t in asv_tokens])

    gen_asvs = np.apply_along_axis(map, -1, gen_tokens.numpy())

    max_asv_count = np.max([len(sample_tokens) for sample_tokens in sample_asvs])

    sample_asvs = [
        np.expand_dims(np.pad(asvs, (0, max_asv_count - len(asvs)), constant_values=""), axis=0) for asvs in sample_asvs
    ]
    sample_asvs = np.vstack(sample_asvs)

    sample_counts = [
        np.expand_dims(np.pad(counts, ((0, max_asv_count - len(counts)), (0, 0))), axis=0) for counts in sample_counts
    ]
    sample_counts = np.vstack(sample_counts)

    assert np.array_equal(sample_asvs, gen_asvs)
    assert np.array_equal(sample_counts, gen_counts)


if __name__ == "__main__":
    test_sample_info()
