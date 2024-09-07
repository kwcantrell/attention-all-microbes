from math import comb

import tensorflow as tf


@tf.function(jit_compile=True)
def _pairwise_distances(embeddings, squared=False):
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
    distances = (
        tf.expand_dims(square_norm, 0)
        - 2.0 * dot_product
        + tf.expand_dims(square_norm, 1)
    )

    # Because of computation errors, some distances might be negative so we
    # put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0
        # (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be
        # exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


class PairwiseLoss(tf.keras.losses.Loss):
    def __init__(self, reduction="none", **kwargs):
        super().__init__(reduction="none", **kwargs)

        @tf.py_function(Tout=tf.float32)
        def nc2(n):
            return tf.cast(comb(n, 2), dtype=tf.float32)

        def pairwise_loss(y_true, y_pred):
            # batch_size = tf.shape(y_pred)[0]
            y_pred_dist = _pairwise_distances(y_pred, squared=False)
            differences = tf.math.square(y_pred_dist - y_true)
            # distances = tf.linalg.band_part(differences, 0, -1)
            # distances = tf.reduce_sum(differences, axis=-1, keepdims=True)
            distances = tf.reduce_sum(differences, axis=-2)
            return distances

        self.loss = tf.function(pairwise_loss)

    def call(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def get_config(self):
        return super().get_config()


def pairwise_residual_mse(batch_size, mean=None, std=None):
    @tf.function(jit_compile=True)
    def sqrt_res(ys):
        r = tf.reshape(ys, shape=(-1, 1)) - tf.reshape(ys, shape=(1, -1))
        r = tf.square(r)
        r_mask = tf.cast(tf.equal(r, 0.0), tf.float32)
        r = r + r_mask * 1e-16
        r = tf.sqrt(r)
        return r * (1.0 - r_mask)

    @tf.function(jit_compile=True)
    def inner(y_true, y_pred):
        if mean is not None:
            y_true = denormalize(y_true, mean, std)
            y_pred = denormalize(y_pred, mean, std)
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        mse = tf.reduce_sum(tf.square(y_true - y_pred)) / batch_size
        r_yt = sqrt_res(y_true)
        r_yp = sqrt_res(y_pred)
        rse = tf.linalg.band_part(tf.square(r_yt - r_yp), 0, -1)
        mrse = tf.reduce_sum(rse) / comb(batch_size, 2)
        return mse + mrse

    return inner


def denormalize(tensor, shift, scale):
    return tensor * scale + shift


def mae_loss(shift=0, scale=1):
    def mae(y_true, y_pred):
        y_true = denormalize(y_true, shift, scale)
        y_pred = denormalize(y_pred, shift, scale)
        return tf.abs(y_true - y_pred)

    return mae


def mse_loss(mean=None, std=None):
    def mse(y_true, y_pred):
        y_true = denormalize(y_true, mean, std)
        y_pred = denormalize(y_pred, mean, std)
        return tf.square(y_true - y_pred)

    return mse


def real_feature_mask(total_features, size):
    total_features = tf.expand_dims(total_features, axis=-1)
    mask = tf.cast(tf.range(size), dtype=tf.int32)
    mask = tf.less(mask, total_features)
    return mask


class MeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.fn = mse_loss(mean, std)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(self.fn(y_true, y_pred), axis=-1)


class PairwiseMSE(tf.keras.losses.Loss):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.fn = mse_loss(mean, std)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(self.fn(y_pred, y_true), axis=-1)
