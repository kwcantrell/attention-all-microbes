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
    def __init__(
        self,
        table: Union[str, Table] = None,
        metadata: Optional[Union[str, pd.DataFrame]] = None,
        metadata_column: Optional[str] = None,
        shuffle: bool = False,
        rarefy_depth: int = 1000,
        epochs: int = 1000,
        gen_new_tables: bool = False,
        batch_size: int = 8,
        is_16S: bool = True,
        gen_new_table_frequency=3,
        return_sample_ids=False,
        tree_path=None,
        seed=None,
    ):
        if isinstance(table, str):
            table = load_table(table)

        self.table: Table = table
        self.tree_path: str = tree_path
        self.metadata_column: str = metadata_column
        self.rarefy_depth: int = rarefy_depth
        self.return_sample_ids: bool = return_sample_ids

        self.shuffle = shuffle
        self.epochs = epochs
        self.gen_new_tables = gen_new_tables
        self.samples_per_minibatch = batch_size

        self.batch_size = batch_size
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

        print("rarefy table...")
        self.metadata: pd.Series = metadata
        self.rarefied_table: Table = self.table.subsample(rarefy_depth)

        self.groups = {}
        for token in self.group_tokens:
            ids = self.metadata[self.metadata == token].index
            self.groups[token] = ids.to_numpy()
            print(f"Group {token} size:", len(self.groups[token]))

        self.size = self.rarefied_table.shape[1]
        self.steps_per_epoch = self.size // self.batch_size
        self.samples_per_group = self.batch_size // len(self.group_tokens)

        if self.samples_per_group * len(self.group_tokens) != self.batch_size:
            raise Exception("Batch size must be a multiple of the number of groups")

        self.y_data = self.metadata.loc[self._rarefied_table.ids()]
        self.on_epoch_end()

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        batch_sample_ids = []
        for group, sample_ids in self.groups.items():
            batch_sample_ids.append(
                np.random.choice(sample_ids, self.samples_per_group, replace=True)
            )
        batch_sample_ids = np.concatenate(batch_sample_ids)
        return self._batch_data(batch_sample_ids)

    def _batch_data(self, batch_sample_ids):
        num_unique_asvs, sparse_indices, obs_indices, counts = [], [], [], []
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

        num_unique_asvs = np.array(num_unique_asvs, dtype=np.int32)
        sparse_indices = np.vstack(sparse_indices, dtype=np.int32)
        obs_indices = np.hstack(obs_indices, dtype=np.int32)
        counts = np.hstack(counts, dtype=np.float32)[:, np.newaxis]

        # get list of unique observations in batch
        unique_obs, obs_indices = np.unique(obs_indices, return_inverse=True)
        lookup = {
            "a": 1,
            "c": 2,
            "g": 3,
            "t": 4,
        }

        def map(asv):
            asv = asv.lower()
            return np.array([lookup[c] for c in asv], dtype=np.int32)[np.newaxis, :]

        tokens = np.concatenate([map(asv) for asv in self.asv_ids[unique_obs]], axis=0)
        y_true = self.y_data.loc[batch_sample_ids].to_numpy()[:, np.newaxis]

        if self.return_sample_ids:
            return (tokens, sparse_indices, obs_indices, counts), batch_sample_ids

        if self.encoder_target is None:
            return (tokens, sparse_indices, obs_indices, counts), y_true

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

        if self.shuffle:
            np.random.shuffle(self.sample_indices)

        self.epochs_since_last_table += 1

    @property
    def rarefied_table(self):
        return self._rarefied_table

    @rarefied_table.setter
    def rarefied_table(self, table: Table):
        self._rarefied_table = table
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
    @add_lock
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
        print(
            "Prefilter shape:",
            metadata.shape,
            metadata[self.metadata_column].value_counts(),
        )

        print("aligning table with metadata")
        table_ids = self.table.ids()
        sample_count_mask = self.table.sum(axis="sample") >= self.rarefy_depth
        samp_ids = np.intersect1d(table_ids[sample_count_mask], metadata.index)
        self.table.filter(samp_ids, axis="sample", inplace=True)
        self.table.remove_empty()
        metadata = metadata.loc[self.table.ids(), [self.metadata_column]]

        print(f"aligned table shape: {self.table.shape}")
        print(f"aligned metadata shape: {metadata.shape}")
        le = preprocessing.LabelEncoder()
        metadata.loc[:, "token"] = le.fit_transform(metadata)
        tokens = metadata["token"]
        print(
            "Postfilter shape:",
            metadata.shape,
            metadata[self.metadata_column].value_counts(),
            tokens.value_counts(),
        )
        self._metadata = tokens.reindex(self.table.ids())
        self.group_tokens = self._metadata.unique()
        print("Triplet group tokens:", self.group_tokens)
        print("done preprocessing metadata")


if __name__ == "__main__":
    ug = TripletGenerator(
        table="/Users/kacantrell/cancer-qiita/tissue_cancer_patient/AAM-study-triplet/triplet_table.biom",
        metadata="/Users/kacantrell/cancer-qiita/tissue_cancer_patient/AAM-study-triplet/triplet-training-metadata.tsv",
        metadata_column="tissue_type",
        gen_new_tables=True,
        batch_size=4,
    )

    x, y = ug[0]
    (tokens, batch_indices, obs_indices, counts) = x
    print("tokens:", tokens.shape)
    print("batch_indices:", batch_indices.shape, batch_indices)
    print("obs indices:", obs_indices.shape)
    print("counts:", counts)
    print("y_true", y)
