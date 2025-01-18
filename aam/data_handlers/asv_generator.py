from __future__ import annotations

import numpy as np
import tensorflow as tf
from bp import parse_newick, to_skbio_treenode

# Unicode mapping dictionary
mapping = {65: 1, 67: 2, 71: 3, 84: 4}  # Maps Unicode numbers to specific values


# Create the mapping function
def map_unicode(val):
    return mapping.get(val, 0)  # Return 0 if the value is not in the mapping


TOKENIZER = tf.keras.layers.TextVectorization(
    max_tokens=6,
    split="character",
    vocabulary=["a", "c", "t", "g"],
    output_mode="int",
    pad_to_max_tokens=True,
    output_sequence_length=150,
)


def tokenize_asv(asv):
    # TextVectorization layer use 0 and 1 for <MASK> and <UNK>
    # AAM expects A to map to 1
    tokens = TOKENIZER(asv)
    mask = tokens > 0
    tokens = tf.cast(tokens, dtype=tf.int32) - tf.cast(mask, dtype=tf.int32)
    return tokens


class ASVGenerator(tf.keras.utils.Sequence):
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
    ):
        tree = parse_newick(open(tree).read())
        self.tree_node = to_skbio_treenode(tree)
        self.return_asv_ids = return_asv_ids

        # extracting ASV tokens
        # step 1: find which nodes represent 150bp ASVs
        print("Data pipeline initialization...")
        print("step 1: find which nodes represent 150bp ASVs")
        asv_preorderpos = []
        obs_encodings = []
        for i, node in enumerate(self.tree_node.preorder(include_self=True)):
            if node.is_tip() and len(node.name) == 150:
                asv_preorderpos.append(i)
                obs_encodings.append(node.name)

        # step 2: extract ASV tokens
        self.asv_preorderpos = np.array(asv_preorderpos, dtype=np.int32)
        self.obs_encodings = np.array(obs_encodings)

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

        self.sample_indices = np.arange(len(self.obs_encodings), dtype=np.int32)
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

    def _sample_data(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = self.obs_encodings[samples]

        if self.return_asv_ids:
            asv_pos = self.asv_preorderpos[samples]
            return tokens, np.array([self.preorder_nodes[i].name for i in asv_pos])

        num_asvs = self.pairwise_batch_size
        samples = samples[:num_asvs]
        asv_pos = self.asv_preorderpos[samples]
        post_order_pos = np.array([self.preorder_nodes[i].postorder_pos for i in asv_pos], dtype=np.int32)
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
                pairwise_distance = (i_to_root - lca_to_root) + (j_to_root - lca_to_root)
                dists[_ri, _rj] = pairwise_distance / self.max_dist_to_root

        return tokens, dists + dists.T


def get_dataset(gen: ASVGenerator):
    def generator():
        for _ in range(1000):
            sequence = np.arange(gen.steps_per_epoch, dtype=np.int32)
            if gen.shuffle:
                np.random.shuffle(sequence)

            for i in sequence:
                yield gen[i]

    if not gen.return_asv_ids:
        y_type = tf.TensorSpec(shape=(gen.pairwise_batch_size, gen.pairwise_batch_size), dtype=tf.float32)
    else:
        y_type = tf.TensorSpec(shape=(gen.samples_per_minibatch), dtype=tf.string)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(tf.TensorSpec(shape=(gen.samples_per_minibatch), dtype=tf.string), y_type),
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    import numpy as np

    tree_path = "/home/kalen/aam-research-exam/research-exam/agp/data/agp-aligned.nwk"
    ug = ASVGenerator(tree=tree_path, sequence_batch_size=128, pairwise_batch_size=32, shuffle=False)

    dataset = get_dataset(ug)
    for x, y in dataset:
        print(x, y)
