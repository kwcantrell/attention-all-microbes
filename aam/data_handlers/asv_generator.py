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
        taxonomy: str = None,
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
        tree = parse_newick(open(tree).read())
        self.tree_node = to_skbio_treenode(tree)
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
        asv_preorderpos = []
        self.obs_ids = []
        obs_encodings = []
        for i, node in enumerate(self.tree_node.preorder(include_self=True)):
            if node.is_tip() and len(node.name) == 150:
                asv_preorderpos.append(i)
                self.obs_ids.append(node.name)
                obs_encodings.append([lookup[c] for c in node.name.lower()])

        # step 2: extract ASV tokens
        self.asv_preorderpos = np.array(asv_preorderpos, dtype=np.int32)
        self.obs_encodings = np.array(obs_encodings, dtype=np.int32)

        # step 3: cache node info
        print("step 3: cache node info")
        self.preorder_nodes = []
        for n in self.tree_node.preorder(include_self=True):
            if n.length is None:
                n.length = 0.0

            if not n.is_root():
                n.length += n.parent.length

            if n.is_tip():
                n.parents = self._root_to_node(n)
            self.preorder_nodes.append(n)

        # cache postorder position to aid with lca computation
        for i, n in enumerate(self.tree_node.postorder(include_self=True)):
            n.postorder_pos = i

        # step 4: find distance from node to root
        print("step 4: find distance from node to root")

        def dist_to_root(preorder_index):
            return self.preorder_nodes[preorder_index].length

        vfunc_dist_to_root = np.vectorize(dist_to_root, otypes=[np.float32])
        self.max_dist_to_root = np.max(vfunc_dist_to_root(self.asv_preorderpos))
        print(f"max tip to root distance is {self.max_dist_to_root}")

        # step 5: find distance from node to lca
        print("step 5: find distance from node to lca")

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
        if taxonomy is not None:
            print("step 6: get taxonomy")
            self.taxonomy = taxonomy
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

    @property
    def taxonomy(self) -> pd.DataFrame:
        return self._taxonomy

    @taxonomy.setter
    def taxonomy(self, taxonomy: Union[str, pd.DataFrame]):
        if taxonomy is None:
            self._taxonomy = taxonomy
            return

        if isinstance(taxonomy, str):
            taxonomy = pd.read_csv(taxonomy, sep="\t", index_col=0)

        taxonomy = taxonomy.loc[self.obs_ids]
        taxonomy[self.levels] = taxonomy[self.taxon_field].str.split("; ", expand=True)

        self.num_tokens = []
        for level in self.levels[1:]:
            level_index = self.levels.index(level)
            levels = self.levels[: level_index + 1]
            tax_level = taxonomy.loc[:, levels]
            taxonomy.loc[:, f"{level} class"] = tax_level.loc[:, levels].agg(
                "; ".join, axis=1
            )

            le = preprocessing.LabelEncoder()
            taxonomy.loc[:, f"{level} token"] = le.fit_transform(
                taxonomy[f"{level} class"]
            )
            taxonomy.loc[:, f"{level} token"] += (
                1  # shifts tokens to be between 1 and n
            )
            self.num_tokens.append(np.max(taxonomy.loc[:, f"{level} token"]) + 1)
        self._taxonomy = taxonomy

    def _root_to_node(self, node):
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

        current_lca = self.tree_node
        for i in range(1, min(l_size, r_size), 1):
            if left_parents[i] != right_parents[i]:
                return current_lca
            current_lca = left_parents[i]

        return current_lca

    def _sample_data(
        self, samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = self.obs_encodings[samples]

        if self.return_asv_ids:
            asv_pos = self.asv_preorderpos[samples]
            return tokens, np.array([self.preorder_nodes[i].name for i in asv_pos])

        tip_tip_samples = samples[: self.pairwise_batch_size]
        num_asvs = len(tip_tip_samples)
        asv_pos = self.asv_preorderpos[tip_tip_samples]
        post_order_pos = np.array(
            [self.preorder_nodes[i].postorder_pos for i in asv_pos], dtype=np.int32
        )
        sorted_post_indx = np.argsort(post_order_pos)

        asv_pos = asv_pos[sorted_post_indx]

        dists = np.zeros((num_asvs, num_asvs))

        for i in range(num_asvs):
            _ri = sorted_post_indx[i]
            pre_i = asv_pos[i]

            # get leaf i
            leaf_i = self.preorder_nodes[pre_i]
            lca = None

            for j in range(i + 1, num_asvs, 1):
                _rj = sorted_post_indx[j]

                # get leaf_j
                pre_j = asv_pos[j]
                leaf_j = self.preorder_nodes[pre_j]

                # get length of i, j, and lca to root
                i_to_root = leaf_i.length
                j_to_root = leaf_j.length

                if lca is None or lca.postorder_pos < leaf_j.postorder_pos:
                    lca = self._lca(leaf_i.parents, leaf_j.parents)

                lca_to_root = lca.length

                # distance from i to j
                pairwise_distance = (i_to_root - lca_to_root) + (
                    j_to_root - lca_to_root
                )
                dists[_ri, _rj] = pairwise_distance / self.max_dist_to_root

        if self.taxonomy is None:
            return tokens, dists + dists.T

        taxonomy = self.taxonomy.loc[[self.obs_ids[i] for i in samples]]
        tax_levels = tuple(
            [taxonomy[f"{level} token"].to_numpy() for level in self.levels[1:]]
        )
        return tokens, (dists + dists.T, tax_levels)


def get_dataset(gen: ASVGenerator):
    enqueuer = tf.keras.utils.OrderedEnqueuer(gen, use_multiprocessing=True)
    enqueuer.start(workers=2, max_queue_size=gen.steps_per_epoch)
    gen.stop = lambda: enqueuer.stop(0.1)

    batch_dim = gen.samples_per_minibatch if gen.drop_remainder else None
    pairwise_batch_size = gen.pairwise_batch_size if gen.drop_remainder else None
    if not gen.return_asv_ids:
        y_type = tf.TensorSpec(
            shape=(pairwise_batch_size, pairwise_batch_size), dtype=tf.float32
        )
    else:
        y_type = tf.TensorSpec(shape=(batch_dim), dtype=tf.string)

    if gen.taxonomy is None:
        dataset = tf.data.Dataset.from_generator(
            enqueuer.get,
            output_signature=(
                tf.TensorSpec(shape=(batch_dim, 150), dtype=tf.int32),
                y_type,
            ),
        )
    else:
        dataset = tf.data.Dataset.from_generator(
            enqueuer.get,
            output_signature=(
                tf.TensorSpec(shape=(batch_dim, 150), dtype=tf.int32),
                (
                    y_type,
                    (
                        tf.TensorSpec(shape=(batch_dim), dtype=tf.int32),
                        tf.TensorSpec(shape=(batch_dim), dtype=tf.int32),
                        tf.TensorSpec(shape=(batch_dim), dtype=tf.int32),
                        tf.TensorSpec(shape=(batch_dim), dtype=tf.int32),
                        tf.TensorSpec(shape=(batch_dim), dtype=tf.int32),
                        tf.TensorSpec(shape=(batch_dim), dtype=tf.int32),
                    ),
                ),
            ),
        )

    return dataset


if __name__ == "__main__":
    import numpy as np

    tree_path = "/home/kalen/aam-research-exam/research-exam/agp/data/agp-aligned.nwk"
    taxonomy_path = (
        "/home/kalen/aam-research-exam/research-exam/agp/data/agp-taxonomy.tsv"
    )
    ug = ASVGenerator(
        tree=tree_path,
        taxonomy=taxonomy_path,
        sequence_batch_size=4,
        pairwise_batch_size=4,
        shuffle=False,
    )
    dataset = get_dataset(ug)
    for x, y in dataset.take(1):
        print(x)
        dist, tokens = y
        print(dist)
        print(tokens)
    print(ug.num_tokens)
