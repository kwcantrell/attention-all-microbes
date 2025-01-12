from __future__ import annotations

import os
from functools import wraps
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from biom import Table, load_table
from bp import parse_newick, to_skbio_treenode


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


def distance_to_parent_node(tree, i, j):
    dist_to_parent = tree.length(i)
    parent = tree.parent(i)
    while parent != j:
        dist_to_parent += tree.length(parent)
        parent = tree.parent(parent)
    return dist_to_parent


def distance_from_i_to_j(tree, i, j):
    lca = tree.lca(i, j)
    if lca == i:
        return distance_to_parent_node(tree, j, i)

    if lca == j:
        return distance_to_parent_node(tree, i, j)

    return distance_to_parent_node(tree, i, lca) + distance_to_parent_node(tree, j, lca)


# Unicode mapping dictionary
mapping = {65: 1, 67: 2, 71: 3, 84: 4}  # Maps Unicode numbers to specific values


# Create the mapping function
def map_unicode(val):
    return mapping.get(val, 0)  # Return 0 if the value is not in the mapping


class ASVGenerator:
    table_fn = "table.biom"
    taxonomy_fn = "taxonomy.tsv"
    axes = np.array(["counts", "tokens", "y", "encoder"])
    # # These are the UTF-8 encodings of A, C, T, G respectively
    # # lookup table converts utf-8 encodings to token
    # # tokens start at 1 to make room for pad token
    lookup_table = np.vectorize(map_unicode)

    def __init__(
        self,
        tree,
        obs_encodings,
        max_tip_root_dist,
        nodes,
        shuffle: bool = False,
        epochs: int = 1000,
        batch_size: int = 8,
        max_bp: int = 150,
        cache=None,
        seed=None,
    ):
        self.tree = tree
        self.obs_encodings = obs_encodings
        self.max_tip_root_dist = max_tip_root_dist
        self.nodes = nodes
        self.shuffle = shuffle
        self.epochs = epochs
        self.samples_per_minibatch = batch_size

        self.batch_size = batch_size
        self.max_bp = max_bp
        self.seed = seed

        self.size = len(self.obs_encodings)
        self.steps_per_epoch = max(self.size // self.batch_size, 1)
        print("Number of sequences:", self.size)

    def _sample_data(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = self.obs_encodings[samples]

        num_asvs = len(samples)
        distances = np.zeros(shape=(num_asvs, num_asvs), dtype=np.float32)
        nodes = self.nodes[samples]
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                dist = distance_from_i_to_j(self.tree, nodes[i], nodes[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return tokens, distances / self.max_tip_root_dist

    def _epoch_complete(self, processed):
        if processed < self.steps_per_epoch:
            return False
        return True

    def _minibatch_indices(self, minibatch, sample_indices):
        start = (minibatch * self.samples_per_minibatch) % len(sample_indices)
        end = (start + self.samples_per_minibatch) % len(sample_indices)
        if start > end:
            start = 0
            end = self.samples_per_minibatch
        return sample_indices[start:end]

    def _epoch_samples(self, sample_indices):
        if self.shuffle:
            print("shuffling...")
            np.random.shuffle(sample_indices)

        return sample_indices

    def _create_epoch_generator(self):
        def generator():
            processed = 0
            sample_indices = np.arange(len(self.obs_encodings))
            for epoch in range(self.epochs):
                print(f"Finished epcoh: {epoch} processed {processed}")
                processed = 0
                minibatch = 0
                sample_indices = self._epoch_samples(sample_indices)

                def sample_data(minibatch):
                    samples = self._minibatch_indices(minibatch, sample_indices)
                    return self._sample_data(samples)

                while not self._epoch_complete(processed):
                    tokens, distances = sample_data(minibatch)

                    tokens = tokens.astype(np.int32)
                    yield (tokens, distances)
                    processed += 1
                    minibatch += 1

        return generator

    def get_data(self):
        generator = self._create_epoch_generator()

        token_sig = tf.TensorSpec(shape=[None, self.max_bp], dtype=tf.int32)
        length_sig = tf.TensorSpec(shape=[None, None], dtype=tf.float32)
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(token_sig, length_sig),
        )

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
    cache = "temp"
    tree = parse_newick(open(tree_path).read())
    # asvs = []
    # nodes = []
    # distance_to_root = []
    # for i in range(tree.B.size):
    #     name = tree.name(i)
    #     if name is not None:
    #         if len(name) == 150:
    #             nodes.append(i)
    #             asvs.append(name)
    #             distance_to_root.append(distance_to_parent_node(tree, i, tree.root()))

    # distance_to_root = np.array(distance_to_root)
    # max_tip_root_dist = np.max(distance_to_root)
    # print(f"found {len(asvs)} in tree and {len(distance_to_root)}, {distance_to_root[:10]}")

    # asvs = asvs[:2048]
    # distance_to_root = distance_to_root[:2048]
    # obs_encodings = np.array([[ord(char) for char in string] for string in asvs])
    # obs_encodings = ASVGenerator.lookup_table(obs_encodings)

    obs_encodings = np.load(f"{cache}-encodings.npy")
    max_tip_root_dist = np.load(f"{cache}-max-tip-root-dist.npy")
    nodes = np.load(f"{cache}-nodes.npy")
    # np.save(f"{cache}-encodings.npy", obs_encodings)
    # np.save(f"{cache}-max-tip-root-dist.npy", max_tip_root_dist)
    # np.save(f"{cache}-nodes.npy", nodes)

    ug = ASVGenerator(tree=tree, obs_encodings=obs_encodings, max_tip_root_dist=max_tip_root_dist, nodes=nodes, batch_size=128)
    data = ug.get_data()
    for i, tokens in enumerate(data["dataset"]):
        print(tokens)
        break
