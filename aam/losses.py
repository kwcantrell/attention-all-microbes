from __future__ import annotations

from typing import Union

import tensorflow as tf


def _pairwise_distances(
    X: tf.Tensor, y: Union[tf.Tensor, None] = None, squared=False
) -> tf.Tensor:
    """Constructs a distance matrix between embedding tensors x and y.

    Args:
        x (tf.Tensor): float tensor of shape [N, E].
        y (Union[tf.Tensor, None], optional): float tensor of shape [M, E]
        squared (bool, optional): If true, computes euclidean distances between between.
        Otherwise, computes dot product between x and y. Defaults to False.

    Returns:
        tf.Tensor: If y is provided returns tensor of shape [N, M]. Otherwise return tensor
        of shape [N,N].
    """
    print("pairwise dist")
    r = tf.reduce_sum(X * X, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    distances = r - 2 * tf.matmul(X, X, transpose_b=True) + tf.transpose(r)
    distances = tf.clip_by_value(distances, 0.0, float("inf"))
    if not squared:
        distances = tf.clip_by_value(distances, 1e-6, float("inf"))
        distances = tf.sqrt(distances)
    return distances


def global_embedding_l2_regulization(sample_embeddings):
    # do not need to take sqrt as ||x|| == ||x||^2 when x is unit length
    norm = tf.reduce_sum(sample_embeddings * sample_embeddings, axis=-1)

    # drive embeddings towards unit length
    return tf.reduce_mean(tf.square(1 - norm))


def _pairwise_cosine_distance(
    x: tf.Tensor, y: Union[tf.Tensor, None] = None
) -> tf.Tensor:
    """Computes the cosine distance between embedding tensors x and y.

    Args:
        x (tf.Tensor): float tensor of shape [N, E].
        y (Union[tf.Tensor, None], optional): float tensor of shape [M, E]

    Returns:
        tf.Tensor: If y is provided returns tensor of shape [N, M]. Otherwise return tensor
        of shape [N,N].
    """
    print("cos dist")
    x = tf.linalg.l2_normalize(x, axis=-1)
    if y is None:
        y = x
    else:
        y = tf.linalg.l2_normalize(y, axis=-1)
    distances = tf.matmul(x, y, transpose_b=True)
    distances = 1 - distances
    distances = tf.clip_by_value(distances, 0.0, 2.0)
    return distances


def global_orthogonal_regulization(sample_embeddings, non_matching_pairs_mask):
    sample_embeddings = tf.cast(sample_embeddings, dtype=tf.float64)
    emb_dim = tf.cast(tf.shape(sample_embeddings)[-1], dtype=tf.float64)
    sample_norm = tf.norm(sample_embeddings, axis=-1, keepdims=True)
    sample_embeddings = tf.divide(sample_embeddings, sample_norm)
    non_matching_pairs_mask = tf.cast(non_matching_pairs_mask, dtype=tf.float64)
    N = tf.reduce_sum(non_matching_pairs_mask)
    N = tf.maximum(tf.cast(1.0, dtype=tf.float64), N)
    sample_inner_prod = tf.matmul(
        sample_embeddings, sample_embeddings, transpose_b=True
    )
    sample_inner_prod = tf.abs(sample_inner_prod * non_matching_pairs_mask)

    # return tf.reduce_sum(sample_inner_prod, axis=-1) / N
    m1 = tf.reduce_sum(sample_inner_prod) / N
    m2 = tf.reduce_sum(tf.square(sample_inner_prod)) / N
    ortho_loss = m1 * m1 + tf.maximum(
        tf.cast(0.0, dtype=tf.float64), m2 - tf.cast(1.0, dtype=tf.float64) / emb_dim
    )
    return tf.cast(ortho_loss, dtype=tf.float32)


class PairwiseLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        loss_type="mse",
        use_mean_pairs=False,
        reduction="none",
        **kwargs,
    ):
        super().__init__(reduction=reduction, **kwargs)
        self.loss_type = loss_type
        self.use_mean_pairs = use_mean_pairs
        if self.loss_type == "mse":
            self.fn = _pairwise_distances
        else:
            print("using cos distance!")
            self.fn = _pairwise_cosine_distance

    def call(self, y_true, y_pred):
        y_pred_dist = self.fn(y_pred)

        differences = tf.math.square(y_true - y_pred_dist)
        mask = tf.linalg.band_part(tf.ones_like(differences), 0, -1)
        mask -= tf.linalg.band_part(mask, 0, 0)
        differences = differences * mask

        loss = tf.reduce_sum(differences) / tf.reduce_sum(mask)

        hard_mask = tf.cast(differences > loss, dtype=tf.float32) * mask
        hard_loss = tf.reduce_sum(differences) / tf.reduce_sum(hard_mask)

        hard_flag = loss < 0.01

        loss = tf.where(hard_flag, hard_loss, loss)
        return [loss, tf.cast(hard_flag, dtype=tf.float32)]


def triplet_loss(embeddings, groups=2, hard_margin=0.025, soft_margin=0.1):
    emb_shape = tf.shape(embeddings, out_type=tf.int32)
    batch_dim = emb_shape[0]
    group_size = tf.math.floordiv(batch_dim, tf.cast(groups, dtype=tf.int32))

    matching_mask = tf.linalg.diag(tf.ones(shape=[group_size]))
    matching_mask = tf.tile(matching_mask, [groups, groups])
    off_diag = 1 - tf.linalg.diag(tf.ones(shape=[batch_dim]))
    non_matching_mask = tf.cast((1 - matching_mask) * off_diag, dtype=tf.bool)
    matching_mask = tf.cast(matching_mask * off_diag, dtype=tf.bool)

    distances = _pairwise_distances(embeddings)

    matching_pairs = tf.expand_dims(distances[matching_mask], axis=-1)
    non_matching_pairs = tf.reshape(distances[non_matching_mask], shape=[batch_dim, -1])

    triplet_loss = (matching_pairs + soft_margin) - non_matching_pairs

    easy_mask = tf.cast(triplet_loss < 0, dtype=tf.float32)
    hard_mask = tf.cast(triplet_loss > (soft_margin - hard_margin), dtype=tf.float32)
    semi_hard_mask = tf.cast(1 - (easy_mask + hard_mask), dtype=tf.bool)

    loss = tf.reduce_mean(triplet_loss[semi_hard_mask])
    return tf.where(loss > 0, loss, 0.0)


def _roll(inputs):
    tensor, shift = inputs
    return tf.roll(tensor, shift=shift, axis=1)


def categorical_triplet_loss(embeddings, num_groups, soft_margin=0.05):
    shape = tf.shape(embeddings)
    batch_dim = shape[0]
    samples_per_group = batch_dim // num_groups
    num_pos_examples_per_sample = samples_per_group - 1
    batch_dim = num_groups * samples_per_group
    group_matching_pairs_mask = tf.ones(
        shape=[samples_per_group, samples_per_group], dtype=tf.int32
    )
    group_matching_pairs_mask = tf.pad(
        group_matching_pairs_mask, paddings=[[0, 0], [0, batch_dim - samples_per_group]]
    )
    matching_pair_mask = tf.reshape(
        tf.tile(group_matching_pairs_mask, multiples=[num_groups, 1]),
        shape=[num_groups, -1, batch_dim],
    )

    matching_pair_mask = tf.map_fn(
        _roll,
        elems=(
            matching_pair_mask,
            tf.range(start=0, limit=num_groups, dtype=tf.int32) * samples_per_group,
        ),
        fn_output_signature=tf.int32,
    )
    matching_pair_mask = tf.reshape(tf.stack(matching_pair_mask), shape=[batch_dim, -1])

    # compute all pairwise distances
    distances = _pairwise_distances(embeddings)

    # extract all distances that are between samples of the same class
    matching_pairs = tf.reshape(
        distances[matching_pair_mask == 1], shape=[batch_dim, samples_per_group]
    )
    matching_pairs = tf.reshape(
        matching_pairs, shape=[num_groups, samples_per_group, samples_per_group]
    )

    # extrall all distances that are between samples of different classes
    non_matching_pairs = tf.reshape(
        distances[(1 - matching_pair_mask) == 1],
        shape=[batch_dim, samples_per_group * (num_groups - 1)],
    )
    non_matching_pairs = tf.reshape(
        non_matching_pairs,
        shape=[num_groups, samples_per_group, (num_groups - 1) * samples_per_group],
    )

    # groups should be orthogal to each other
    ortho_loss = global_orthogonal_regulization(
        embeddings, (1 - matching_pair_mask) == 1
    )

    def _group_triplet_loss(inputs):
        """Computes all triplets for a given class."""
        group_dist, non_group_dist = inputs

        # remove the group distances that represent the distance from a sample to itself
        group_pair_mask = (
            1 - tf.linalg.diag(tf.ones(samples_per_group, dtype=tf.int32))
        ) > 0
        group_dist = tf.reshape(
            group_dist[group_pair_mask],
            shape=[samples_per_group, samples_per_group - 1, 1],
        )
        group_dist = tf.transpose(group_dist, perm=[1, 0, 2])
        group_dist = group_dist - tf.expand_dims(non_group_dist, axis=0)
        return tf.transpose(group_dist, perm=[1, 0, 2])

    triplets = tf.map_fn(
        _group_triplet_loss,
        (matching_pairs, non_matching_pairs),
        fn_output_signature=tf.float32,
    )
    triplets = tf.reshape(
        triplets,
        shape=[
            num_groups * samples_per_group,
            num_pos_examples_per_sample,
            samples_per_group * (num_groups - 1),
        ],
    )

    # find semi-hard triplets
    non_hard_mask = tf.cast(triplets > 0, dtype=tf.float32)
    semi_hard_mask = tf.cast(triplets <= soft_margin, dtype=tf.float32)
    triplet_mask = non_hard_mask * semi_hard_mask

    # compute loss per positive example in group
    per_sample_loss = tf.reduce_sum(triplets * triplet_mask, axis=-1)
    per_sample_loss = tf.math.divide_no_nan(
        per_sample_loss, tf.reduce_sum(triplet_mask, axis=-1)
    )

    # compute loss across entire group for each sample
    return tf.reduce_mean(per_sample_loss, axis=-1) * 0.0, ortho_loss
