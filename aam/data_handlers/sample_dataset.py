from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table


class SequenceEmbeddings:
    def __init__(self, model, embeddings_fp, labels_fp, normalize=False):
        emb = np.load(embeddings_fp, allow_pickle=True)
        self.embeddings = emb
        print(self.embeddings.shape)
        if "aam" in model:
            print("aam embeddings")
        else:
            print("normalizing embeddings")
        self.normalize()

        self.labels = labels_fp
        print(len(self.labels))

    def iget(self, indices):
        return self.embeddings[indices]

    def filter(self, keep):
        labels, indices, _ = np.intersect1d(
            self.labels, keep, assume_unique=True, return_indices=True
        )
        if len(indices) != len(keep):
            print("We have a problem!!!!")
        self.embeddings = self.embeddings[indices]
        self._labels = self._labels[indices]

    def get(self, ids):
        labels, indices, _ = np.intersect1d(
            self.labels, ids, assume_unique=True, return_indices=True
        )
        return self.embeddings[indices]

    def normalize(self):
        emb_mean = np.mean(self.embeddings, axis=-1, keepdims=True)
        emb_std = np.std(self.embeddings, axis=-1, keepdims=True)
        self.embeddings = (self.embeddings - emb_mean) / emb_std

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, fp):
        labels = np.load(fp, allow_pickle=True)
        self._labels = labels.astype(np.str_)

    def __getitem__(self, indices):
        return self.iget(indices)

    def filter_table(self, table: Table):
        table_asvs = table.ids(axis="observation")
        keep = set(table_asvs).intersection(set(self.labels))
        return table.filter(keep, axis="observation", inplace=False)

    def align_table(self, table: Table):
        """adds missing sequences and reorders table so that sequences are
        in same order as self.labels"""

        # add asvs that were dropped from the table during rarefaction
        table_asvs = set(table.ids(axis="observation"))
        missing_asv = set(self.labels).difference(table_asvs)
        dropped_table = Table(
            np.zeros((len(missing_asv), table.shape[1])),
            list(missing_asv),
            table.ids(),
        )
        table = table.concat(dropped_table, axis="observation")
        return table.sort_order(self.labels, axis="observation")


class SampleDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        model,
        table: Union[str, Table] = None,
        metadata: Optional[Union[str, pd.DataFrame]] = None,
        metadata_column: Optional[str] = None,
        sequence_embeddings: Optional[str] = None,
        sequence_labels: Optional[str] = None,
        normalize_embeddings: bool = False,
        shuffle: bool = False,
        rarefied_table=None,
        epochs: int = 1000,
        gen_new_tables: bool = False,
        gen_new_table_frequency=3,
        return_sample_ids=False,
        seed=None,
        shuffle_ranks=True,
        insert_random_sequence=False,
        batch_size=64,
        max_member_taxa=175,
    ):
        if isinstance(table, str):
            table = load_table(table)

        self.max_member_taxa = max_member_taxa
        self.table: Table = table
        self.sequence_embeddings = SequenceEmbeddings(
            model, sequence_embeddings, sequence_labels, normalize_embeddings
        )
        self.sequence_embeddings.filter(self.table.ids(axis="observation"))
        self.table = self.sequence_embeddings.filter_table(self.table)
        self.table = self.sequence_embeddings.align_table(self.table)
        self.asv_indices = np.arange(self.table.shape[0], dtype=np.int32)
        count_weights = self.table.pa(inplace=False).sum(axis="observation")
        count_weights = count_weights / count_weights.sum()
        self.sorted_count_indices = np.argsort(count_weights)[::-1]
        self.count_weights = (count_weights > 0) / (count_weights > 0).sum()
        self.sorted_count_indices = self.sorted_count_indices[
            : self.max_member_taxa
        ]

        self.metadata_column: str = metadata_column
        self.metadata: pd.Series = metadata
        self.return_sample_ids: bool = return_sample_ids

        self.shuffle = shuffle
        self.shuffle_ranks = shuffle_ranks
        self.insert_random_sequences = insert_random_sequence
        self.epochs = epochs
        self.gen_new_tables = gen_new_tables

        self.seed = seed
        self.gen_new_table_frequency = gen_new_table_frequency
        self.epochs_since_last_table = 0

        print("rarefy table...")
        if rarefied_table is not None:
            self.rarefied_table: Table = rarefied_table
        else:
            self.rarefied_table = self.table.copy()
        self.sample_ids = self.rarefied_table.ids()
        self.size = self.rarefied_table.shape[1]
        self.batch_size = batch_size
        if self.batch_size > len(self.sample_ids):
            self.steps_per_epoch = 1
        else:
            self.steps_per_epoch = self.size // self.batch_size
        self.random_state = np.random.default_rng(2021)

    def on_epoch_end(self):
        self.random_state.shuffle(self.sample_ids)

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        return self._batch_data(self.sample_ids[start:end])

    def _gen_random_set(self, exclude, nsamples):
        vs = self.random_state.choice(
            self.asv_indices, nsamples, p=self.count_weights, replace=False
        )
        return np.setdiff1d(vs, exclude)

    def _batch_data(self, batch_sample_ids):
        embeddings = []
        attention_masks = []
        true_memberships = []
        for i, sample_id in enumerate(batch_sample_ids):
            full_sample_counts = self.table.data(sample_id, dense=True)
            rarefied_sample_data = self.rarefied_table.data(
                sample_id, dense=False
            ).tocoo()
            (obs_idx, _), sample_counts = (
                rarefied_sample_data.coords,
                rarefied_sample_data.data,
            )

            count_mask = sample_counts > 0
            obs_idx = obs_idx[count_mask]
            _sample_indices = obs_idx[count_mask]
            _sample_counts = sample_counts[count_mask]
            # total_true_members = min(len(_sample_counts), self.max_member_taxa)
            # base_indices = np.arange(len(_sample_indices))
            # base_indices = self.random_state.choice(
            #     base_indices,
            #     size=total_true_members,
            #     p=_sample_counts / _sample_counts.sum(),
            #     replace=False,
            # )
            # if self.shuffle_ranks:
            #     num_sorted = max(1, int(total_true_members * 0.90))
            # else:
            #     num_sorted = total_true_members
            # self.random_state.shuffle(base_indices)

            # sort = base_indices[:num_sorted]
            # sort = sort[np.argsort(_sample_counts[sort])]
            # sort = sort[::-1]

            # if self.shuffle_ranks:
            #     randomize = base_indices[num_sorted:total_true_members]
            #     random_insert = self.random_state.choice(
            #         np.arange(len(sort), dtype=np.int32),
            #         size=len(randomize),
            #         replace=True,
            #     )
            #     true_indices = np.insert(sort, random_insert, randomize)
            # else:
            #     true_indices = sort
            # _sample_indices = _sample_indices[true_indices]
            # _sample_counts = _sample_counts[true_indices]

            # if self.insert_random_sequences:
            #     random_indices = self._gen_random_set(
            #         obs_idx, max(1, int(total_true_members * 0.15))
            #     )
            #     random_insert = self.random_state.choice(
            #         np.arange(len(true_indices), dtype=np.int32),
            #         size=len(random_indices),
            #         # p=_sample_counts / _sample_counts.sum(),
            #         replace=False,
            #     )
            #     _sample_indices = np.insert(
            #         _sample_indices, random_insert, random_indices
            #     )
            #     _sample_counts = np.insert(_sample_counts, random_insert, 0)

            total_true_members = min(len(_sample_indices), self.max_member_taxa)
            if total_true_members < len(_sample_indices):
                _sample_indices = self.random_state.choice(
                    _sample_indices,
                    size=total_true_members,
                    p=_sample_counts / _sample_counts.sum(),
                    replace=False,
                )
            nadd = self.max_member_taxa - len(_sample_indices)
            if nadd > 0:
                if self.insert_random_sequences:
                    # missing_indices = self.random_state.choice(
                    #     self.asv_indices,
                    #     size=nadd,
                    #     p=self.count_weights,
                    # )
                    missing_indices = self._gen_random_set(
                        _sample_indices, self.max_member_taxa
                    )
                else:
                    missing_indices = np.setdiff1d(
                        self.sorted_count_indices, _sample_indices
                    )
                    missing_indices = missing_indices[
                        np.argsort(self.count_weights[missing_indices])
                    ]
                _sample_indices = np.hstack(
                    [_sample_indices, missing_indices[:nadd]]
                )
            _sample_indices = _sample_indices[np.argsort(_sample_indices)]
            _sample_counts = full_sample_counts[_sample_indices]

            # max size is self.
            embeddings.append(
                self.sequence_embeddings.embeddings[_sample_indices]
            )
            attention_masks.append([[1.0]] * len(_sample_indices))
            true_memberships.append(_sample_counts)
        # pad embeddings
        max_seq_dim = max([len(es) for es in embeddings])

        embeddings = [
            np.pad(es, ((0, max_seq_dim - len(es)), (0, 0)), constant_values=0)
            for es in embeddings
        ]
        embeddings = np.stack(embeddings).astype(np.float32)
        # pad attention masks
        attention_masks = [
            np.pad(
                mask,
                ((0, max_seq_dim - len(mask)), (0, 0)),
                constant_values=0.0,
            )
            for mask in attention_masks
        ]
        attention_masks = np.stack(attention_masks).astype(np.float32)
        # pad membership output
        true_memberships = [
            np.pad(mask, (0, max_seq_dim - len(mask)), constant_values=0)
            for mask in true_memberships
        ]
        true_memberships = np.stack(true_memberships).astype(np.int32)

        if self.metadata is None:
            return (embeddings, attention_masks, true_memberships), 0

        # need to add sparse padding!!!
        y_true = self.metadata.loc[batch_sample_ids]
        y_true = y_true.reindex(batch_sample_ids)
        return (embeddings, attention_masks, true_memberships), y_true

    @property
    def rarefied_table(self):
        return self._rarefied_table

    @rarefied_table.setter
    def rarefied_table(self, rarefied_table: Table):
        print("removing empty sample/obs from table")
        if rarefied_table is None:
            self._rarefied_table = None
            return
        rarefied_table.remove_empty()
        self._rarefied_table = self.sequence_embeddings.align_table(
            rarefied_table
        )

    def rarefy_table(self, table):
        rarefied_table = table.subsample(self.rarefy_depth)
        rarefied_table.remove_empty()
        rarefied_table = self.sequence_embeddings.align_table(rarefied_table)
        return rarefied_table

    @property
    def metadata(self) -> pd.Series:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        if metadata is None:
            self._metadata = None
            return
        if isinstance(metadata, str):
            metadata = pd.read_csv(
                metadata, sep="\t", index_col=0, dtype={0: str}
            )
        if self.metadata_column not in metadata.columns:
            raise Exception(f"Invalid metadata column {self.metadata_column}")
        print("aligning table with metadata")
        samp_ids = np.intersect1d(self.table.ids(axis="sample"), metadata.index)
        self.table.filter(samp_ids, axis="sample", inplace=True)
        self.table.remove_empty()
        metadata = metadata.loc[self.table.ids(), [self.metadata_column]]
        print(f"aligned table shape: {self.table.shape}")
        print(f"aligned metadata shape: {metadata.shape}")
        metadata = metadata.astype(np.int32)
        self._metadata = metadata.reindex(self.table.ids())
        print("done preprocessing metadata")
