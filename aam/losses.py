from __future__ import annotations

from typing import Union

import tensorflow as tf


def _pairwise_distances_unstable(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean
        distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of
    # `dot_product`.
    # This also provides more numerical stability (the diagonal of the result
    # will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we
    # put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

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


def _pairwise_distances(x: tf.Tensor, y: Union[tf.Tensor, None] = None, squared=False):
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


def _pairwise_cosine_distances(x: tf.Tensor, y: Union[tf.Tensor, None] = None, squared=False):
    if y is None:
        y = x
    cos_sim = tf.matmul(x, y, transpose_b=True)
    return 1 - cos_sim


def global_orthogonal_regulization(sample_embeddings, non_matching_pairs_mask):
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

        # extract just the upper triangle of distance matrix
        # mask = tf.linalg.band_part(y_true, 0, -1) > 0
        # mask = tf.reshape(mask, shape=[-1])
        # differences = tf.reshape(differences, shape=[-1])[mask]

        # get hardest example
        return tf.reduce_max(differences)


def triplet_loss(embeddings, groups=2, margin=0.2):
    emb_shape = tf.shape(embeddings, out_type=tf.int32)
    batch_dim = emb_shape[0]
    group_size = batch_dim // tf.cast(groups, dtype=tf.int32)

    matching_mask = tf.linalg.diag(tf.ones(shape=[group_size]))
    matching_mask = tf.tile(matching_mask, [groups, groups])
    off_diag = 1 - tf.linalg.diag(tf.ones(shape=[batch_dim]))
    non_matching_mask = tf.cast((1 - matching_mask) * off_diag, dtype=tf.bool)
    matching_mask = tf.cast(matching_mask * off_diag, dtype=tf.bool)

    distances = _pairwise_cosine_distances(embeddings)

    matching_pairs = tf.expand_dims(distances[matching_mask], axis=-1)
    non_matching_pairs = tf.reshape(distances[non_matching_mask], shape=[batch_dim, -1])

    triplet_loss = matching_pairs - non_matching_pairs + margin
    valid_mask = tf.cast(triplet_loss > 0, dtype=tf.float32)
    triplet_loss = triplet_loss * valid_mask

    hard_mask = tf.cast(non_matching_pairs < matching_pairs, dtype=tf.float32)
    semi_hard_mask = tf.cast(non_matching_pairs < matching_pairs + margin, dtype=tf.float32) * (1 - hard_mask)

    semi_hard_loss = tf.reduce_mean(triplet_loss[tf.cast(semi_hard_mask, dtype=tf.bool)])

    return semi_hard_loss + 0.1 * global_orthogonal_regulization(embeddings, non_matching_mask)


@tf.keras.saving.register_keras_serializable(package="ImbalancedCategoricalCrossEntrop")
class ImbalancedMSE(tf.keras.losses.Loss):
    def __init__(self, max_density, reduction="none", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.max_density = max_density

    def call(self, y_true, y_pred):
        y, density = y_true
        y = tf.ensure_shape(y, [None, 1])
        y_pred = tf.ensure_shape(y_pred, [None, 1])
        loss = tf.square(y - y_pred)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({"max_density": self.max_density})
        return config


@tf.keras.saving.register_keras_serializable(package="ImbalancedCategoricalCrossEntrop")
class ImbalancedCategoricalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, adjustment_weights=[0.1, 0.2, 0.3], reduction="none", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.num_classes = len(adjustment_weights)
        adjustment_weights = tf.constant(adjustment_weights)
        adjustment_weights = tf.reduce_sum(adjustment_weights) / adjustment_weights
        adjustment_weights = tf.expand_dims(adjustment_weights, axis=-1)
        self.adjustment_weights = adjustment_weights

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.int32)
        weights = tf.nn.embedding_lookup(self.adjustment_weights, y_true)
        weights = tf.reshape(weights, shape=[-1])

        y_true = tf.one_hot(y_true, self.num_classes)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return (weights) * loss

    def get_config(self):
        config = super().get_config()
        config.update({"adjustment_weights": self.adjustment_weights})
        return config
