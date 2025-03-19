from __future__ import annotations

from functools import wraps
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table
from bp import parse_newick, to_skbio_treenode
from sklearn import preprocessing


def add_lock(func):
    lock = f"_{func.__name__}_lock"

    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        if not hasattr(obj, lock):
            setattr(obj, lock, True)
            return func(obj, *args, **kwargs)

        if getattr(obj, lock):
            raise Exception(f"Attempting to modify locked property '{func.__name__}'")

        setattr(obj, lock, True)
        return func(obj, *args, **kwargs)

    return wrapper


class TripletGenerator(tf.keras.utils.Sequence):
    taxon_field = "Taxon"
    levels = [f"Level {i}" for i in range(1, 8)]

    def __init__(
        self,
        table: Union[str, Table] = None,
        taxonomy: Optional[Union[str, pd.DataFrame]] = None,
        metadata: Optional[Union[str, pd.DataFrame]] = None,
        metadata_column: Optional[str] = None,
        sequence_embeddings: Optional[str] = None,
        sequence_labels: Optional[str] = None,
        shift: Optional[Union[str, float]] = None,
        scale: Union[str, float] = "minmax",
        max_token_per_sample: int = 1024,
        shuffle: bool = False,
        rarefy_depth: int = 1000,
        epochs: int = 1000,
        gen_new_tables: bool = False,
        samples_per_group: int = 8,
        max_bp: int = 150,
        is_16S: bool = True,
        is_categorical: Optional[bool] = None,
        gen_new_table_frequency=3,
        return_sample_ids=False,
        tree_path=None,
        steps_per_epoch=100,
        max_groups=5,
        seed=None,
        drop_remainder=True,
        batch_size=128,
        upsample=True,
    ):
        if isinstance(table, str):
            table = load_table(table)

        self.table: Table = table
        self.tree_path = tree_path
        self.is_categorical: bool = is_categorical
        self.metadata_column: str = metadata_column
        self.shift = shift
        self.scale = scale
        self.metadata: pd.Series = metadata
        self.rarefy_depth: int = rarefy_depth
        self.max_token_per_sample: int = max_token_per_sample
        self.return_sample_ids: bool = return_sample_ids
        self.max_groups = max_groups

        self.sequence_embeddings = sequence_embeddings
        self.sequence_labels = sequence_labels
        if self.sequence_embeddings is not None:
            sequence_embeddings = np.load(self.sequence_embeddings)
            emb_mean = np.mean(sequence_embeddings, axis=0)
            emb_std = np.std(sequence_embeddings, axis=0)
            self.sequence_embeddings = (sequence_embeddings - emb_mean) / (
                emb_std + 1e-8
            )

            self.sequence_labels = np.load(self.sequence_labels, allow_pickle=True)
            self.sequence_labels = self.sequence_labels.astype(np.str_)
            self.sequence_labels = np.char.encode(
                self.sequence_labels, encoding="utf-8"
            )
            print(self.sequence_labels.dtype)

        self.include_sample_weight: bool = is_categorical

        self.shuffle = shuffle
        self.epochs = epochs
        self.gen_new_tables = gen_new_tables

        self.max_bp = max_bp
        self.is_16S = is_16S
        self.seed = seed
        self.gen_new_table_frequency = gen_new_table_frequency
        self.epochs_since_last_table = 0

        self.encoder_target = None
        self.encoder_dtype = None
        self.encoder_output_type = None
        self.sample_ids = None
        self.asv_ids = None

        if self.tree_path is not None:
            self.tree = to_skbio_treenode(parse_newick(open(self.tree_path).read()))
            self.postorder_pos = {
                n.name: i for i, n in enumerate(self.tree.postorder()) if n.is_tip()
            }

        self.tax_level = f"Level {1}"
        self.taxonomy = taxonomy
        if taxonomy is not None:
            print("taxonomy info", self.num_tax_values)

        print("rarefy table...")
        self.rarefied_table: Table = self.table.subsample(rarefy_depth, seed=42)
        self.size = self.rarefied_table.shape[1]
        self.groups = self.metadata[self.metadata_column].unique()
        self.num_groups = len(self.groups)
        self.samples_per_group = samples_per_group
        self.groups_per_step = self.num_groups  # min(self.max_groups, self.num_groups)

        le = preprocessing.LabelEncoder()
        self._metadata = self.metadata.loc[self._rarefied_table.ids()]
        groups = self.metadata[self.metadata_column]
        y_data = le.fit_transform(groups)
        self.y_data = pd.Series(y_data, groups.index)
        self.on_epoch_end()

        self.upsample = upsample
        if self.upsample:
            self.batch_size = self.groups_per_step * self.samples_per_group
            self.samples_per_minibatch = self.batch_size
            self.steps_per_epoch = steps_per_epoch
        else:
            self.drop_remainder = drop_remainder
            self.batch_size = batch_size
            self.steps_per_epoch = self.size // self.batch_size
            self.sample_ids = self._rarefied_table.ids()
            if (
                not self.drop_remainder
                and self.steps_per_epoch * self.batch_size < self.size
            ):
                self.steps_per_epoch += 1

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        if self.upsample:
            batch_sample_ids = []
            metadata = self.metadata.loc[self.y_data.index, self.metadata_column]
            groups = metadata.unique()
            counts = np.repeat(
                metadata.value_counts().to_numpy()[:, np.newaxis],
                repeats=self.samples_per_group,
                axis=1,
            )
            totals = counts.sum(axis=0)
            weights = counts / totals[np.newaxis, :]
            weights = weights.reshape((-1))
            for group in groups:
                ids = metadata[metadata == group].index.to_numpy()
                batch_sample_ids.append(
                    np.random.choice(ids, self.samples_per_group, replace=True)
                )
                # ids = group[
                #     idx * self.samples_per_group : (idx + 1) * self.samples_per_group
                # ]
                # if len(ids) == self.samples_per_group:
                #     batch_sample_ids.append(ids)
            return self._batch_data(np.hstack(batch_sample_ids), weights)
        else:
            start = idx * self.batch_size
            end = start + self.batch_size
            return self._batch_data(self.sample_ids[start:end], 1.0)

    def _batch_data(self, batch_sample_ids, weights):
        num_unique_asvs, sparse_indices, obs_indices, counts, taxon_counts = (
            [],
            [],
            [],
            [],
            [],
        )
        cur_row_indx = 0
        for s_id in batch_sample_ids:
            sample_data = self.rarefied_table.data(s_id, dense=False).tocoo()
            (obs_idx, _), sample_counts = sample_data.coords, sample_data.data

            # remove zeros
            non_zero_mask = sample_counts > 0.0
            obs_idx = obs_idx[non_zero_mask]
            sample_counts = sample_counts[non_zero_mask]

            num_unique_asvs.append(len(obs_idx))
            sparse_indices.append([[cur_row_indx, i] for i in range(len(obs_idx))])

            obs_indices.append(obs_idx)
            counts.append(sample_counts)
            cur_row_indx += 1

            if self.taxonomy is not None:
                taxons = self.taxonomy.loc[self.asv_ids[obs_idx], "Taxon"].to_numpy()
                sample_tax_counts = np.zeros(self.num_tax_values)
                np.add.at(sample_tax_counts, taxons, sample_counts)
                taxon_counts.append(sample_tax_counts)

        num_unique_asvs = np.array(num_unique_asvs, dtype=np.int32)
        sparse_indices = np.vstack(sparse_indices, dtype=np.int32)
        obs_indices = np.hstack(obs_indices, dtype=np.int32)
        counts = np.hstack(counts, dtype=np.float32)[:, np.newaxis]

        if self.taxonomy is not None:
            taxon_counts = np.vstack(taxon_counts, dtype=np.float32)

        # get list of unique observations in batch
        unique_obs, obs_indices = np.unique(obs_indices, return_inverse=True)
        if self.sequence_embeddings is None:
            lookup = {
                "a": 1,
                "c": 2,
                "g": 3,
                "t": 4,
            }

            def map(asv):
                asv = asv.lower()
                return np.array([lookup[c] for c in asv], dtype=np.int32)[np.newaxis, :]

            tokens = np.concatenate(
                [map(asv) for asv in self.asv_ids[unique_obs]], axis=0
            )
        else:
            asvs, asv_ids_idx, sequence_labels_idx = np.intersect1d(
                self.asv_ids[unique_obs],
                self.sequence_labels,
                assume_unique=True,
                return_indices=True,
            )
            tokens = self.sequence_embeddings[sequence_labels_idx]
        y_true = self.y_data.loc[batch_sample_ids].to_numpy()[:, np.newaxis]

        if self.return_sample_ids:
            return (tokens, sparse_indices, obs_indices, counts), batch_sample_ids

        if self.encoder_target is None:
            if self.taxonomy is None:
                return (tokens, sparse_indices, obs_indices, counts), y_true
            # y_age = self.metadata.loc[
            #     self.metadata.index.isin(batch_sample_ids), "host_age_normalized_years"
            # ]

            return (
                tokens,
                sparse_indices,
                obs_indices,
                counts,
                taxon_counts,
            ), y_true  # (y_true, y_age.to_numpy()[:, np.newaxis] / 100.0)

        encoder_output = self._encoder_output(batch_sample_ids)
        return (tokens, sparse_indices, obs_indices, counts), (y_true, encoder_output)

    def on_epoch_end(self):
        if (
            self.gen_new_tables
            and self.epochs_since_last_table > self.gen_new_table_frequency
        ):
            print("resampling dataset...")
            self.rarefied_table = self.table.subsample(self.rarefy_depth)
            self.epochs_since_last_table = 0

        # self.group_ids = []
        # for group, group_df in self.metadata.groupby(self.metadata_column):
        #     self.group_ids.append(list(group_df.index))
        # if self.shuffle:
        #     for group in self.group_ids:
        #         np.random.shuffle(group)

        if self.shuffle:
            np.random.shuffle(self.sample_ids)

        self.epochs_since_last_table += 1

    @property
    def taxonomy(self):
        return self._taxonomy

    @taxonomy.setter
    def taxonomy(self, taxonomy):
        if taxonomy is None:
            self._taxonomy = taxonomy
            return

        taxonomy[self.levels] = taxonomy[self.taxon_field].str.split("; ", expand=True)
        taxonomy = taxonomy.loc[taxonomy[self.tax_level].str.len() > 3]
        taxonomy = taxonomy.loc[:, self.levels]
        taxonomy.loc[:, "Taxon"] = taxonomy.loc[:, self.levels[:6]].agg(
            "; ".join, axis=1
        )
        self.table = self.table.filter(
            set(self.table.ids(axis="observation")).intersection(set(taxonomy.index)),
            axis="observation",
        )
        self.table.remove_empty()

        print(taxonomy["Taxon"].to_numpy())

        le = preprocessing.LabelEncoder()
        taxonomy["Taxon"] = le.fit_transform(taxonomy.loc[taxonomy.index, "Taxon"])
        self.taxonomy_values = taxonomy["Taxon"].unique()
        self.num_tax_values = np.max(self.taxonomy_values) + 1
        self._taxonomy = taxonomy

    @property
    def rarefied_table(self):
        return self._rarefied_table

    @rarefied_table.setter
    def rarefied_table(self, table: Table):
        self._rarefied_table = table
        self._metadata = self._metadata.loc[self._rarefied_table.ids()]
        print("removing empty sample/obs from table")
        self._rarefied_table.remove_empty()
        if self.tree_path is not None:

            def sort_obs(obs):
                post_pos = [self.postorder_pos[ob] for ob in obs]
                sorted_indices = np.argsort(post_pos)
                return obs[sorted_indices]

            self._rarefied_table = self._rarefied_table.sort(
                sort_obs, axis="observation"
            )

        self.sample_ids = self._rarefied_table.ids()
        self.asv_ids = self._rarefied_table.ids(axis="observation")
        self.sample_indices = np.arange(len(self.sample_ids))

        print("creating encoder target...")
        self.encoder_target = self._create_encoder_target()
        print("encoder target created")

    def _create_encoder_target(self) -> None:
        return None

    def _encoder_output(self, sample_ids):
        return None

    @property
    def table(self) -> Table:
        return self._table

    @table.setter
    def table(self, table: Union[str, Table]):
        self._table = table

    @property
    def metadata(self) -> pd.Series:
        return self._metadata

    @metadata.setter
    @add_lock
    def metadata(self, metadata: Union[str, pd.DataFrame]):
        if metadata is None:
            return

        if isinstance(metadata, str):
            metadata = pd.read_csv(metadata, sep="\t", index_col=0, dtype={0: str})

        if self.metadata_column not in metadata.columns:
            raise Exception(f"Invalid metadata column {self.metadata_column}")

        print("aligning table with metadata")
        samp_ids = np.intersect1d(self.table.ids(axis="sample"), metadata.index)
        self.table.filter(samp_ids, axis="sample", inplace=True)
        self.table.remove_empty()
        metadata = metadata.loc[self.table.ids()]
        print(f"aligned table shape: {self.table.shape}")
        print(f"aligned metadata shape: {metadata.shape}")
        self._metadata = metadata.reindex(self.table.ids())
        print("done preprocessing metadata")


# def get_dataset(gen: TripletGenerator):
#     enqueuer = tf.keras.utils.OrderedEnqueuer(gen, use_multiprocessing=True)
#     enqueuer.start(workers=2, max_queue_size=gen.steps_per_epoch)
#     gen.stop = lambda: enqueuer.stop(0.1)

#     batch_dim = gen.samples_per_minibatch
#     if not gen.return_sample_ids:
#         y_type = tf.TensorSpec(shape=(batch_dim, 1), dtype=tf.string)
#     else:
#         y_type = tf.TensorSpec(shape=(batch_dim), dtype=tf.string)

#     dataset = tf.data.Dataset.from_generator(
#         enqueuer.get,
#         output_signature=(
#             (
#                 tf.TensorSpec(shape=[None, 150], dtype=tf.int32),
#                 tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
#                 tf.TensorSpec(shape=[None], dtype=tf.int32),
#                 tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
#                 tf.TensorSpec(
#                     shape=[gen.batch_size, gen.num_tax_values], dtype=tf.float32
#                 ),
#             ),
#             y_type,
#         ),
#     )
#     return dataset


if __name__ == "__main__":
    from aam.models.triplet_encoder import TripletEncoder

    taxonomy = pd.read_csv(
        "/home/kalen/removing-study-id/agp-unique-samples-taxonomy.tsv",
        sep="\t",
        index_col=0,
    )
    # taxonomy = taxonomy.loc[taxonomy["Taxon"].str.len() > 3]

    ug = TripletGenerator(
        table="/home/kalen/removing-study-id/agp-unique-samples.biom",
        metadata="/home/kalen/removing-study-id/sg-train.tsv",
        metadata_column="sequence_group",
        taxonomy=taxonomy,
        sequence_embeddings="/home/kalen/removing-study-id/sg-train-asv-embeddings.npy",
        sequence_labels="/home/kalen/removing-study-id/sg-train-asv-labels.npy",
        gen_new_tables=True,
        samples_per_group=2,
        max_groups=10,
        shuffle=True,
    )
    x, y1 = ug[0]
    # x, y2 = ug[ug.steps_per_epoch + 1]
    print(y1)
    # print(x, y, y.shape)
    # # x, y = ug[0]
    # # print(y)
    # # print(ug.num_groups)
    # dataset = get_dataset(ug)
    # # (tokens, batch_indices, obs_indices, counts) = x
    # # print("tokens:", tokens.shape)
    # # print("batch_indices:", batch_indices.shape, batch_indices)
    # # print("obs indices:", obs_indices.shape)
    # # print("counts:", counts)
    # # print("y_true", y)

    # asv_encoder = tf.keras.models.load_model(
    #     "/home/kalen/aam-research-exam/research-exam/healty-age-regression/unifrac-encoder-large/model.keras",
    #     compile=False,
    # )
    # model = TripletEncoder(asv_encoder)

    # token_shape = tf.TensorShape([None, 150])
    # batch_indicies = tf.TensorShape([None, 2])
    # indicies_shape = tf.TensorShape([None])
    # count_shape = tf.TensorShape([None, 1])
    # taxonomy_count = tf.TensorShape([None, ug.num_tax_values])
    # model.build(
    #     [token_shape, batch_indicies, indicies_shape, count_shape, taxonomy_count]
    # )
    # for x, y in dataset:
    #     print(model(x))
    # # print(model(x))
