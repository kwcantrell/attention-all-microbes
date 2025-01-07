from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import tensorflow as tf
from biom import Table, load_table
from bp import parse_newick
from skbio import DistanceMatrix

from aam.data_handlers.unifrac_generator import UniFracGenerator


class GOTUGenerator(UniFracGenerator):
    def __init__(
        self,
        table: Union[str, Table],
        asv_table: Union[str, Table],
        asv_rarefy_depth: int = 5000,
        **kwargs,
    ):
        if isinstance(table, str):
            table = load_table(table)
        if isinstance(asv_table, str):
            asv_table = load_table(asv_table)
        gotu_samples = table.ids()
        asv_samples = asv_table.ids()
        keep = set(gotu_samples).intersection(asv_samples)
        table = table.filter(keep)
        asv_table = asv_table.filter(keep)

        super(GOTUGenerator, self).__init__(table=table, **kwargs)
        self.token_mapping = self.map_tokens()
        kwargs["table"] = asv_table
        kwargs["rarefy_depth"] = asv_rarefy_depth
        kwargs["is_16S"] = True
        self.asv_generator = UniFracGenerator(**kwargs)
        print(len(self.rarefy_table.ids()))
        (
            self.rarefy_table,
            self.table_data,
            self.sample_mask,
            self.asv_generator.rarefy_table,
            self.asv_generator.table_data,
            self.asv_generator.sample_mask,
        ) = self.rarefy_gotu_and_asv_table(self.rarefy_table, self.asv_generator.rarefy_table, rarefy=False)

        valid_mask = self.sample_mask & self.asv_generator.sample_mask
        self.sample_indices = np.arange(len(self.rarefy_table.ids()))
        self.size = len(self.sample_indices)
        self.sample_indices = self.sample_indices[valid_mask]
        self.steps_per_epoch = self.size // self.batch_size

    def rarefy_gotu_and_asv_table(self, gotu_table, asv_table, rarefy=True):
        if rarefy:
            gotu_table = gotu_table.subsample(self.rarefy_depth, seed=self.seed)
            asv_table = asv_table.subsample(self.asv_generator.rarefy_depth, seed=self.asv_generator.seed)

        keep = set(gotu_table.ids()).intersection(asv_table.ids())
        gotu_table = gotu_table.filter(keep)
        asv_table = asv_table.filter(keep)

        gotu_sample_mask = gotu_table.pa(inplace=False).sum(axis="sample") <= self.max_token_per_sample
        asv_sample_mask = asv_table.pa(inplace=False).sum(axis="sample") <= self.asv_generator.max_token_per_sample

        gotu_table_data = self._create_table_data(gotu_table)
        asv_table_data = self.asv_generator._create_table_data(asv_table)
        return (
            gotu_table,
            gotu_table_data,
            gotu_sample_mask,
            asv_table,
            asv_table_data,
            asv_sample_mask,
        )

    def get_data(self, include_sample_ids=False):
        generator = self._create_epoch_generator()
        output_sig = (
            tf.TensorSpec(shape=[self.batch_size], dtype=tf.int32),
            tf.TensorSpec(shape=[None, self.max_bp], dtype=tf.int32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            tf.TensorSpec(shape=[self.batch_size], dtype=tf.int32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
            tf.TensorSpec(shape=[self.batch_size, self.batch_size], dtype=tf.float32),
        )

        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_sig,
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        data_obj = {
            "dataset": dataset,
            "size": self.size,
            "steps_pre_epoch": self.steps_per_epoch,
        }
        return data_obj

    def _create_encoder_target(self, table: Table) -> DistanceMatrix:
        pass

    def _epoch_samples(
        self,
        epoch,
        gotu_table_data,
        gotu_sample_mask,
        asv_table_data,
        asv_sample_mask,
        asv_encoder_target,
        sample_indices,
    ):
        if self.gen_new_tables and epoch > 0:
            print(f"epcoh {epoch}: generating new table...")
            (
                gotu_rarefy_table,
                gotu_table_data,
                gotu_sample_mask,
                asv_rarefy_table,
                asv_table_data,
                asv_sample_mask,
            ) = self.rarefy_gotu_and_asv_table(self.preprocessed_table, self.asv_generator.preprocessed_table)

            asv_encoder_target = self.asv_generator._create_encoder_target(asv_rarefy_table)

            sample_indices = np.arange(len(gotu_rarefy_table.ids()))
            valid_mask = gotu_sample_mask & asv_sample_mask
            sample_indices = sample_indices[valid_mask]

        if self.shuffle:
            print("shuffling...")
            np.random.shuffle(sample_indices)

        return (
            gotu_table_data,
            gotu_sample_mask,
            asv_table_data,
            asv_sample_mask,
            asv_encoder_target,
            sample_indices,
        )

    def _sample_data(
        self,
        gotu_samples: np.ndarray,
        gotu_table_data=None,
        asv_table_data=None,
        asv_encoder_target=None,
    ):
        if gotu_table_data is None:
            gotu_table_data = self.table_data
        if asv_table_data is None:
            asv_table_data = self.asv_generator.table_data

        (
            gotu_row,
            gotu_col,
            gotu_counts,
            gotu_obs_encodings,
            gotu_sample_ids,
            gotu_obs_ids,
        ) = gotu_table_data
        (
            asv_row,
            asv_col,
            asv_counts,
            asv_obs_encodings,
            asv_sample_ids,
            asv_obs_ids,
        ) = asv_table_data

        gotu_s_ids = [gotu_sample_ids[s] for s in gotu_samples]
        asv_sample_mask = np.isin(asv_sample_ids, gotu_s_ids)
        asv_samples = self.asv_generator.sample_indices[asv_sample_mask]

        gotu_row = gotu_row.reshape((1, -1))
        gotu_samples = gotu_samples.reshape((-1, 1))

        asv_row = asv_row.reshape((1, -1))
        asv_samples = asv_samples.reshape((-1, 1))

        gotu_batch_mask = gotu_samples == gotu_row
        gotu_batch_counts = np.sum(gotu_batch_mask.astype(np.int32), axis=-1)
        gotu_mask = np.logical_or.reduce(gotu_batch_mask, axis=0)

        asv_batch_mask = asv_samples == asv_row
        asv_batch_counts = np.sum(asv_batch_mask, axis=-1)
        asv_mask = np.logical_or.reduce(asv_batch_mask, axis=0)

        gotu_s_counts = gotu_counts[gotu_mask]
        asv_s_counts = asv_counts[asv_mask]

        gotu_s_obs = gotu_col[gotu_mask]
        gotu_s_tokens = self.gotu_tokens(gotu_s_obs)

        asv_s_obs = asv_col[asv_mask]
        asv_unique_obj, asv_obj_indices = np.unique(asv_s_obs, return_inverse=True)
        asv_s_tokens = asv_obs_encodings[asv_unique_obj]

        if asv_encoder_target is None:
            asv_encoder_target = self.asv_generator.encoder_target
        asv_encoder_output = self.asv_generator._encoder_output(asv_encoder_target, gotu_s_ids, None)

        return (
            asv_batch_counts,
            asv_s_counts.reshape((-1, 1)),
            asv_s_tokens,
            asv_obj_indices,
            gotu_batch_counts,
            gotu_s_counts.reshape((-1, 1)),
            gotu_s_tokens,
            asv_encoder_output,
        )

    def _create_epoch_generator(self):
        def generator():
            processed = 0
            gotu_table_data = self.table_data
            gotu_sample_mask = self.sample_mask
            asv_table_data = self.asv_generator.table_data
            asv_sample_mask = self.asv_generator.sample_mask
            asv_encoder_target = self.asv_generator.encoder_target

            sample_indices = self.sample_indices
            print(len(sample_indices))
            for epoch in range(self.epochs):
                print(f"Finished epcoh: {epoch} processed {processed}")
                processed = 0
                minibatch = 0
                (
                    gotu_table_data,
                    gotu_sample_mask,
                    asv_table_data,
                    asv_sample_mask,
                    asv_encoder_target,
                    sample_indices,
                ) = self._epoch_samples(
                    epoch,
                    gotu_table_data,
                    gotu_sample_mask,
                    asv_table_data,
                    asv_sample_mask,
                    asv_encoder_target,
                    sample_indices,
                )

                def sample_data(minibatch):
                    samples = self._minibatch_indices(minibatch, sample_indices)
                    return self._sample_data(samples, gotu_table_data, asv_table_data, asv_encoder_target)

                while not self._epoch_complete(processed):
                    (
                        asv_batch_counts,
                        asv_counts,
                        asv_tokens,
                        asv_obj_indices,
                        gotu_batch_counts,
                        gotu_counts,
                        gotu_tokens,
                        asv_encoder_out,
                    ) = sample_data(minibatch)

                    if gotu_counts is not None:
                        processed += 1
                        output = (
                            asv_batch_counts.astype(np.int32),
                            asv_tokens.astype(np.int32),
                            asv_obj_indices.astype(np.int32),
                            asv_counts.astype(np.int32),
                            gotu_batch_counts.astype(np.int32),
                            gotu_tokens.astype(np.int32),
                            gotu_counts.astype(np.int32),
                            asv_encoder_out,
                        )

                        yield output
                    minibatch += 1

        return generator

    def _encoder_output(
        self,
        encoder_target: DistanceMatrix,
        sample_ids: Iterable[str],
        ob_ids: list[str],
    ) -> np.ndarray[float]:
        return super(GOTUGenerator, self)._encoder_output(encoder_target, sample_ids, ob_ids)

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
        token_mapping = {
            "PAD": 0,
            "START": 1,
            "END": 2,
        }
        token_id = 3
        for i in range(1, ((bp_tree.__len__()) * 2)):
            if not bp_tree.isleaf(i):
                continue
            if bp_tree.name(i) is not None:
                node_name = bp_tree.name(i)
                if len(node_name) == 10:
                    token_mapping[node_name] = token_id
                    token_id += 1

        return token_mapping

    def gotu_tokens(self, indicies: np.ndarray) -> np.ndarray:
        obs_ids = self.obs_ids[indicies]
        return np.array([self.token_mapping[id] for id in obs_ids]).reshape((-1, 1))


if __name__ == "__main__":
    gotu_gen = GOTUGenerator(
        table="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/gotu_ordered_table.biom",
        asv_table="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/asv_ordered_table.biom",
        tree_path="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/tulsa-tree.nwk",
        metadata="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/metag_metadata.tsv",
        metadata_column="host_age",
        shift=0.0,
        scale=100.0,
        gen_new_tables=True,
        is_16S=False,
        rarefy_depth=100000,
        asv_rarefy_depth=10000,
    )
    print("Finished creating generator")
    gotu_data = gotu_gen.get_data(include_sample_ids=True)
    gotu_dataset = gotu_data["dataset"]

    # ug = UniFracGenerator(
    #     table="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/asv_ordered_table.biom",
    #     tree_path="/home/jokirkland/data/trees/2022.10.phylogeny.asv.nwk",
    #     metadata="/home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/metag_metadata.tsv",
    #     metadata_column="Age",
    #     shift=0.0,
    #     scale=100.0,
    #     gen_new_tables=True,
    # )
    # asv_data = ug.get_data()
    # asv_dataset = asv_data["dataset"]

    for item in gotu_dataset:
        print(len(item), item)
        break

    # for i, (
    #     asv_batch_counts,
    #     asv_tokens,
    #     asv_indicies,
    #     asv_counts,
    #     gotu_batch_counts,
    #     gotu_tokens,
    #     gotu_counts,
    #     asv_unifrac,
    # ) in enumerate(gotu_dataset):
    #     print("GOTU_DATSET_VALUES", i)
        # print(asv_tokens, asv_counts)
        # print(gotu_tokens, gotu_counts)
        # print(f"Printing X Data: {x}\nPrinting Y Data {y}")
    # full_dataset = tf.data.Dataset.zip((asv_dataset, gotu_dataset))

    # for i, (x, y) in enumerate(full_dataset):
    #     print("FULL MERGED DATASET")
    #     print(f"Printing X Data: {x}\nPrinting Y Data {y}")
    #     break
