from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from bp import parse_newick, to_skbio_treenode
from sklearn import preprocessing


class ASVGenerator(tf.keras.utils.Sequence):
    levels = [f"Level {i}" for i in range(1, 8)]
    taxon_field = "Taxon"

    def __init__(
        self,
        tree,
        shuffle: bool = False,
        epochs: int = 1000,
        sequence_batch_size: int = 8,
        pairwise_batch_size: int = 8,
        max_bp: int = 150,
        subsample: float = 1.0,
        seed=None,
        return_asv_ids=False,
        drop_remainder=True,
    ):
        self.tree_node = to_skbio_treenode(parse_newick(open(tree).read()))
        self.return_asv_ids = return_asv_ids
        self.drop_remainder = drop_remainder

        # extracting ASV tokens
        # step 1: find which nodes represent 150bp ASVs
        print("Data pipeline initialization...")
        print("step 1: find which nodes represent 150bp ASVs")
        lookup = {
            "a": 1,
            "c": 2,
            "g": 3,
            "t": 4,
        }
        self.obs_ids = []
        obs_encodings = []
        self.postorder_nodes = []
        for i, node in enumerate(self.tree_node.postorder(include_self=True)):
            node.postorder_pos = i
            if node.is_tip() and len(node.name) == 150:
                self.obs_ids.append(node.name)
                obs_encodings.append([lookup[c] for c in node.name.lower()])
                self.postorder_nodes.append(node)

        # step 2: extract ASV tokens
        self.obs_encodings = np.array(obs_encodings, dtype=np.int32)

        # step 3: cache node info
        print("step 3: cache node info")
        self.max_dist_to_root = 0
        for n in self.tree_node.preorder(include_self=True):
            if n.length is None:
                n.length = 0.0

            if not n.is_root():
                n.length += n.parent.length

            dist_to_root = n.length
            if dist_to_root > self.max_dist_to_root:
                self.max_dist_to_root = dist_to_root

            if n.is_tip():
                n.parents = self._node_to_root(n)
        print(f"max tip to root distance is {self.max_dist_to_root}")

        self.shuffle = shuffle
        self.epochs = epochs
        self.samples_per_minibatch = sequence_batch_size
        self.pairwise_batch_size = pairwise_batch_size
        self.max_bp = max_bp
        self.seed = seed

        self.size = int(len(self.obs_encodings) * subsample)
        self.steps_per_epoch = max(self.size // self.samples_per_minibatch, 1)
        if (
            not self.drop_remainder
            and self.steps_per_epoch * self.samples_per_minibatch < self.size
        ):
            self.steps_per_epoch += 1
        self.sample_indices = np.arange(len(self.obs_encodings), dtype=np.int32)
        self.num_tokens = None
        self.on_epoch_end()
        print("Number of sequences:", self.size)

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        start = idx * self.samples_per_minibatch
        end = start + self.samples_per_minibatch
        samples = self.sample_indices[start:end]
        return self._sample_data(samples)

    def on_epoch_end(self):
        print("Epoch finished")
        print("Preparing next epoch")
        if self.shuffle:
            np.random.shuffle(self.sample_indices)

    def _node_to_root(self, node):
        parent = node.parent
        parents = []
        while parent != self.tree_node:
            parents.append(parent)
            parent = parent.parent
        parents.append(parent)
        return parents[::-1]

    def _lca(self, left_parents, right_parents):
        l_size = len(left_parents)
        r_size = len(right_parents)

        current_lca = left_parents[0]
        for i in range(1, min(l_size, r_size), 1):
            if left_parents[i] != right_parents[i]:
                return current_lca
            current_lca = left_parents[i]

        return current_lca

    def _sample_data(self, asvs):
        if self.return_asv_ids:
            tokens = self.obs_encodings[asvs]
            return tokens, np.array([self.postorder_nodes[i].name for i in asvs])

        num_asvs = len(asvs)
        sorted_asv_indices = np.argsort(asvs)
        sorted_asvs = [asvs[i] for i in sorted_asv_indices]

        nodes = [self.postorder_nodes[i] for i in sorted_asvs]
        tokens = self.obs_encodings[sorted_asvs]

        # initialize distance matrix
        dists = np.zeros((num_asvs, num_asvs))
        asv_0_dist = nodes[0].length
        dists[0, 1:] += asv_0_dist
        asv_n_dist = nodes[-1].length
        dists[: num_asvs - 1, num_asvs - 1] += asv_n_dist
        for i in range(1, num_asvs - 1, 1):
            asv_i = nodes[i]
            dist_to_root = asv_i.length
            dists[i, i + 1 :] += dist_to_root
            dists[:i, i] += dist_to_root

        for i in range(num_asvs):
            # get leaf i
            leaf_i = nodes[i]

            for j in range(i + 1, num_asvs, 1):
                leaf_j = nodes[j]

                if j == i + 1:
                    cur_lca = self._lca(leaf_i.parents, leaf_j.parents)
                    starting_lca_pos = j

                if cur_lca.postorder_pos < leaf_j.postorder_pos:
                    dists[i, starting_lca_pos:j] -= 2 * cur_lca.length
                    cur_lca = self._lca(leaf_i.parents, leaf_j.parents)
                    starting_lca_pos = j

                if j == num_asvs - 1:
                    dists[i, starting_lca_pos:] -= 2 * cur_lca.length

        return tokens, dists * (1 / self.max_dist_to_root)


if __name__ == "__main__":
    import numpy as np

    tree_path = (
        "/home/kalen/aam-research-exam/research-exam/agp/results/reference-tree.nwk"
    )
    ug = ASVGenerator(
        tree=tree_path,
        sequence_batch_size=4,
        pairwise_batch_size=4,
        shuffle=False,
        return_asv_ids=True,
    )
    print(ug[0])
    # dataset = get_dataset(ug)
    # model = tf.keras.models.load_model(
    #     "/home/kalen/aam-research-exam/research-exam/healty-age-regression/asv-encoder-tax-v4/model.keras",
    #     compile=False,
    # )
    # print(model.predict(dataset.take(1)))
