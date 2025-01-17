from __future__ import annotations

import numpy as np
import tensorflow as tf
from bp import parse_newick, to_skbio_treenode

# Unicode mapping dictionary
mapping = {65: 1, 67: 2, 71: 3, 84: 4}  # Maps Unicode numbers to specific values


# Create the mapping function
def map_unicode(val):
    return mapping.get(val, 0)  # Return 0 if the value is not in the mapping


class ASVGenerator:
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
    ):
        tree = parse_newick(open(tree).read())
        self.tree_node = to_skbio_treenode(tree)

        # extracting ASV tokens
        # step 1: find which nodes represent 150bp ASVs
        print("Data pipeline initialization...")
        print("step 1: find which nodes represent 150bp ASVs")
        preoderposition = np.arange(0, int(tree.B.size / 2), 1, dtype=np.int32)

        def is_150bp(pre_pos):
            name = tree.name(tree.preorderselect(pre_pos))
            if name is not None:
                if len(name) == 150:
                    return True
                return False
            return False

        vfunc_is_150bp = np.vectorize(lambda x: is_150bp(x), otypes=[bool])
        is_150bp_mask = vfunc_is_150bp(preoderposition)
        print(f"found {np.sum(is_150bp_mask)} ASVs")
        asv_preorder = preoderposition[is_150bp_mask]

        # step 2: extract ASV tokens
        print("step 2: extract ASV tokens")
        mapping = {65: 1, 67: 2, 71: 3, 84: 4}

        def get_tokens(pre_pos):
            asv = tree.name(tree.preorderselect(pre_pos))
            return np.array([mapping[ord(c)] for c in asv], dtype=np.int32)

        vfunc_tokens = np.vectorize(get_tokens, otypes=[np.int32], signature="()->(n)")

        self.obs_encodings = vfunc_tokens(asv_preorder)
        self.asv_preorderpos = asv_preorder

        # step 3: cache node info
        print("step 3: cache node info")
        self.preorder_nodes = []
        for n in self.tree_node.preorder(include_self=True):
            if n.length is None:
                n.length = 0.0

            if not n.is_root():
                n.length += n.parent.length
            self.preorder_nodes.append(n)

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
        print("Number of sequences:", self.size)

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

    def _sample_data(self, samples: np.ndarray, return_asv_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = self.obs_encodings[samples]

        asv_pos = self.asv_preorderpos[samples]
        num_asvs = self.pairwise_batch_size

        dists = np.zeros((num_asvs, num_asvs))
        parents = [self._root_to_node(self.preorder_nodes[asv_pos[i]]) for i in range(num_asvs)]

        def pair_dist(i):
            pre_i = asv_pos[i]

            # get leaf i
            leaf_i = self.preorder_nodes[pre_i]

            for j in range(i + 1, num_asvs, 1):
                # get leaf_j
                pre_j = asv_pos[j]
                leaf_j = self.preorder_nodes[pre_j]

                # get length of i, j, and lca to root
                i_to_root = leaf_i.length
                j_to_root = leaf_j.length
                lca_to_root = self._lca(parents[i], parents[j]).length

                # distance from i to j
                pairwise_distance = (i_to_root - lca_to_root) + (j_to_root - lca_to_root)
                dists[i, j] = pairwise_distance / self.max_dist_to_root

        vfunc_pair_dist = np.vectorize(pair_dist, otypes=None)

        # get preorder position of asvs
        vfunc_pair_dist(np.arange(num_asvs, dtype=np.int32))
        if not return_asv_ids:
            return tokens, dists
        else:
            return tokens, np.array([self.preorder_nodes[i].name for i in asv_pos])

    def _epoch_complete(self, processed):
        if processed < self.steps_per_epoch:
            return False
        return True

    def _minibatch_indices(self, minibatch, sample_indices):
        start = minibatch * self.samples_per_minibatch
        end = start + self.samples_per_minibatch
        return sample_indices[start:end]

    def _create_epoch_generator(self, return_asv_ids):
        def generator():
            for epoch in range(self.epochs):
                print(f"Starting epcoh: {epoch}")
                processed = 0
                minibatch = 0
                sample_indices = np.arange(len(self.obs_encodings), dtype=np.int32)

                if self.shuffle:
                    print("shuffling...")
                    np.random.shuffle(sample_indices)

                while not self._epoch_complete(processed):
                    samples = self._minibatch_indices(minibatch, sample_indices)
                    processed += 1
                    minibatch += 1

                    yield self._sample_data(samples, return_asv_ids)

        return generator

    def get_data(self, return_asv_ids=False):
        generator = self._create_epoch_generator(return_asv_ids)

        token_sig = tf.TensorSpec(shape=[None, self.max_bp], dtype=tf.int32)
        if not return_asv_ids:
            pair_dist_sig = tf.TensorSpec(shape=[None, None], dtype=tf.float32)
        else:
            pair_dist_sig = tf.TensorSpec(shape=[None], dtype=tf.string)
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(generator, output_signature=(token_sig, pair_dist_sig))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        data_obj = {
            "dataset": dataset,
            "size": self.size,
            "steps_pre_epoch": self.steps_per_epoch,
        }
        return data_obj


if __name__ == "__main__":
    import numpy as np

    from aam.data_handlers import ASVGenerator

    tree_path = "/home/kalen/aam-research-exam/research-exam/agp/data/agp-aligned.nwk"

    ug = ASVGenerator(tree=tree_path, sequence_batch_size=128, pairwise_batch_size=32, shuffle=False)
    data = ug.get_data(return_asv_ids=False)
    for x in data["dataset"].take(1):
        print(x)
