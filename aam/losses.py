from __future__ import annotations

from typing import Union

import tensorflow as tf


def _pairwise_distances(x: tf.Tensor, y: Union[tf.Tensor, None] = None, squared=False) -> tf.Tensor:
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
    if y is None:
        y = x
    distances = tf.expand_dims(x, axis=0) - tf.expand_dims(y, axis=1)
    distances = tf.multiply(distances, distances)
    distances = tf.reduce_sum(distances, axis=-1)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0
        # (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
        distances = distances + mask * 1e-07

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be
        # exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def global_embedding_l2_regulization(sample_embeddings):
    # do not need to take sqrt as ||x|| == ||x||^2 when x is unit length
    norm = tf.reduce_sum(sample_embeddings * sample_embeddings, axis=-1)

    # drive embeddings towards unit length
    return tf.reduce_mean(tf.square(1 - norm))


def _pairwise_cosine_distance(x: tf.Tensor, y: Union[tf.Tensor, None] = None) -> tf.Tensor:
    """Computes the cosine distance between embedding tensors x and y.

    Args:
        x (tf.Tensor): float tensor of shape [N, E].
        y (Union[tf.Tensor, None], optional): float tensor of shape [M, E]

    Returns:
        tf.Tensor: If y is provided returns tensor of shape [N, M]. Otherwise return tensor
        of shape [N,N].
    """
    x = tf.linalg.l2_normalize(x, axis=-1)
    if y is None:
        y = x
    else:
        y = tf.linalg.l2_normalize(y, axis=-1)
    distances = tf.matmul(x, y, transpose_b=True)
    return 1 - distances


def global_orthogonal_regulization(sample_embeddings, non_matching_pairs_mask):
    sample_embeddings = tf.linalg.l2_normalize(sample_embeddings, axis=-1)
    d = tf.cast(tf.shape(sample_embeddings)[-1], dtype=tf.float32)
    sample_inner_prod = tf.matmul(sample_embeddings, sample_embeddings, transpose_b=True)
    non_matching_pairs = sample_inner_prod[non_matching_pairs_mask]

    m1 = tf.reduce_mean(non_matching_pairs)
    m2 = tf.reduce_mean(tf.square(non_matching_pairs))
    return m1 * m1 + tf.maximum(0.0, m2 - 1 / d)


class PairwiseLoss(tf.keras.losses.Loss):
    def __init__(self, loss_type="mse", reduction="none", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.loss_type = loss_type

    def call(self, y_true, y_pred):
        y_pred_dist = _pairwise_distances(y_pred, squared=False)

        if self.loss_type == "mse":
            differences = tf.math.square(y_pred_dist - y_true)
        elif self.loss_type == "msle":
            differences = tf.math.square(tf.math.log1p(y_pred_dist) - tf.math.log1p(y_true))

        batch_dim = tf.shape(y_true)[0]
        valid_pairs = tf.linalg.band_part(tf.ones_like(differences), 0, -1) - tf.linalg.diag(tf.ones(shape=[batch_dim])) > 0
        valid_differences = differences[valid_pairs]

        return valid_differences


def triplet_loss(embeddings, groups=2, hard_margin=0.0, soft_margin=0.1):
    emb_shape = tf.shape(embeddings, out_type=tf.int32)
    batch_dim = emb_shape[0]
    group_size = batch_dim // tf.cast(groups, dtype=tf.int32)

    matching_mask = tf.linalg.diag(tf.ones(shape=[group_size]))
    matching_mask = tf.tile(matching_mask, [groups, groups])
    off_diag = 1 - tf.linalg.diag(tf.ones(shape=[batch_dim]))
    non_matching_mask = tf.cast((1 - matching_mask) * off_diag, dtype=tf.bool)
    matching_mask = tf.cast(matching_mask * off_diag, dtype=tf.bool)

    distances = _pairwise_cosine_distance(embeddings)

    matching_pairs = tf.expand_dims(distances[matching_mask], axis=-1)
    non_matching_pairs = tf.reshape(distances[non_matching_mask], shape=[batch_dim, -1])

    triplet_loss = (matching_pairs + soft_margin) - non_matching_pairs
    valid_mask = tf.cast(triplet_loss > 0, dtype=tf.float32)
    triplet_loss = triplet_loss * valid_mask

    hard_mask = tf.cast(non_matching_pairs < matching_pairs + hard_margin, dtype=tf.float32)
    semi_hard_mask = tf.cast(non_matching_pairs < matching_pairs + soft_margin, dtype=tf.float32) * (1 - hard_mask)
    semi_hard_loss = triplet_loss[tf.cast(semi_hard_mask, tf.bool)]

    mean = tf.reduce_mean(semi_hard_loss)
    std = tf.math.reduce_std(semi_hard_loss)
    l_std = mean - std
    r_std = mean + std
    mask = (semi_hard_loss >= l_std) & (semi_hard_loss <= r_std)

    loss = tf.reduce_mean(semi_hard_loss[mask])
    return tf.where(loss > 0, loss, 0.0)
