import numpy as np
from biom import Table


class SequenceEmbeddings:
    def __init__(self, embeddings_fp, labels_fp, normalize=False):
        self.embeddings = np.load(embeddings_fp)
        emb_mean = np.mean(self.embeddings, axis=0)
        emb_std = np.std(self.embeddings, axis=0)
        self.embeddings = (self.embeddings - emb_mean) / emb_std

        self.labels = labels_fp
        print(len(self.labels))

    def iget(self, indices):
        return self.embeddings[indices]

    def get(self, ids):
        labels, indices, _ = np.intersect1d(
            self.labels, ids, assume_unique=True, return_indices=True
        )
        return self.embeddings[indices]

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
