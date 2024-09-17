import tensorflow as tf


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
        distances = distances + mask * 1e-07

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be
        # exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


class PairwiseLoss(tf.keras.losses.Loss):
    def __init__(self, reduction="none", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def call(self, y_true, y_pred):
        y_pred_dist = _pairwise_distances(y_pred, squared=False)
        differences = tf.math.square(y_pred_dist - y_true)
        return differences


@tf.keras.saving.register_keras_serializable(package="ImbalancedCategoricalCrossEntrop")
class ImbalancedMSE(tf.keras.losses.Loss):
    def __init__(self, max_density, reduction="none", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.max_density = max_density

    def call(self, y_true, y_pred):
        y, density = y_true
        loss = tf.keras.losses.mse(y, y_pred)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"max_density": self.max_density})
        return config


@tf.keras.saving.register_keras_serializable(package="ImbalancedCategoricalCrossEntrop")
class ImbalancedCategoricalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, adjustment_weights=[0.1, 0.2, 0.3], reduction="none", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        adjustment_weights = tf.constant(adjustment_weights)
        adjustment_weights = tf.reduce_sum(adjustment_weights) / adjustment_weights
        adjustment_weights = tf.expand_dims(adjustment_weights, axis=-1)
        self.adjustment_weights = adjustment_weights
        self.num_classes = len(adjustment_weights)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.int32)
        weights = tf.nn.embedding_lookup(self.adjustment_weights, y_true)
        weights = tf.reshape(weights, shape=[-1])

        y_true = tf.one_hot(y_true, self.num_classes)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return loss * weights

    def get_config(self):
        config = super().get_config()
        config.update({"adjustment_weights": self.adjustment_weights})
        return config
