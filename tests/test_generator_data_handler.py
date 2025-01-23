import numpy as np
import tensorflow as tf
from biom import load_table

from aam.data_handlers.generator_dataset import GeneratorDataset, batch_embeddings

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


def test_sample_info():
    table_path = "agp-no-duplicate-host-bloom-filtered-5000-small-stool-only-very-small.biom"
    metadata_path = "agp-healthy.txt"
    batch_size = 8
    gd = GeneratorDataset(
        table=table_path,
        metadata=metadata_path,
        metadata_column="host_age",
        scale="minmax",
        gen_new_tables=True,
        batch_size=batch_size,
        shuffle=False,
    )

    table = gd.rarefied_table.copy()
    sample_ids = table.ids()
    o_ids = table.ids(axis="observation")

    def _extract_sample(s_id):
        table_data = table.data(s_id, dense=True)
        mask = table_data > 0

        sample_ids, sample_counts = o_ids[mask], table_data[mask]
        sorted_indices = np.argsort(sample_counts)
        sorted_descending = sorted_indices[::-1]
        return sample_ids[sorted_descending], sample_counts[sorted_descending]

    for i in range(len(gd)):
        print(f"testing batch {i}")
        batch_ids = sample_ids[i * batch_size : (i + 1) * batch_size]
        table_batch = [_extract_sample(s_id) for s_id in batch_ids]

        (tokens, indices, asv_indices, counts), y = gd[i]
        batch_tokens, batch_counts = batch_embeddings(tokens, asv_indices, indices, counts)
        batch_gen = []
        for sample, counts in zip(list(batch_tokens.numpy()), list(batch_counts.numpy())):
            sample_asvs = []
            sample_mask = counts.reshape((-1)) > 0
            sample = sample[sample_mask]

            for asv_tokens in sample:
                sample_asvs.append(tokens_to_asv(asv_tokens))
            batch_gen.append((np.hstack(sample_asvs), counts[sample_mask]))

        for table_sample, gen_sample in zip(table_batch, batch_gen):
            table_obs, table_counts = table_sample
            gen_obs, gen_counts = gen_sample

            assert np.array_equal(table_counts.reshape((-1)), gen_counts.reshape((-1)))
            assert np.array_equal(table_obs, gen_obs)


# def test_batch_reconstruction():


if __name__ == "__main__":
    test_sample_info()
