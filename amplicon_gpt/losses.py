import tensorflow as tf

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

@tf.keras.saving.register_keras_serializable(package="Scale16s", name="unifrac_loss_var")
def unifrac_loss_var(y_true, y_pred):
    y_pred_dist = _pairwise_distances(y_pred)
    difference = y_pred_dist - y_true
    square_dist = tf.square(difference)
    var_dist = tf.math.reduce_sum(square_dist, axis=0)
    return tf.reduce_sum(var_dist) / 16.0

@tf.keras.saving.register_keras_serializable(package="Scale16s", name="regression_loss_variance")
def regression_loss_variance(y_true, y_pred):
    true_mean = tf.reduce_mean(y_true)
    true_std = tf.math.reduce_std(y_true)
    pred_mean = tf.reduce_mean(y_pred)
    pred_std = tf.math.reduce_std(y_pred)
    y_true_centered = (y_true - true_mean) / true_std
    y_pred_centered = (y_pred - pred_mean) / pred_std
    xs = tf.square(y_true_centered - y_pred_centered) 
    difference_norm = tf.reduce_sum(tf.square((y_true / true_mean) - (y_pred / true_mean)))
    return tf.math.reduce_mean(xs) + difference_norm

@tf.keras.saving.register_keras_serializable(package="Scale16s", name="regression_loss_difference_in_means")
def regression_loss_difference_in_means(y_true, y_pred):
    true_mean = tf.reduce_mean(y_true)
    pred_mean = tf.reduce_mean(y_pred)

    true_variance = tf.math.reduce_variance(y_true)
    pred_variance = tf.math.reduce_variance(y_pred)
    
    sum_var = pred_variance / 16.0 + true_variance / 16.0
    mask = tf.cast(tf.equal(sum_var, 0.0), tf.float32)
    sum_var = sum_var + mask * 1e-16
    sum_var = tf.sqrt(sum_var)
    # Correct the epsilon added: set the distances on the mask to be exactly 0.0
    sum_var = sum_var * (1.0 - mask)

    t = (pred_mean - true_mean) / sum_var

    return tf.abs(t)

@tf.keras.saving.register_keras_serializable(package="Scale16s", name="regression_loss_difference_in_means")
def regression_loss_combined(y_true, y_pred):
    return  0.2*regression_loss_variance(y_true, y_pred) + regression_loss_difference_in_means(y_true, y_pred)

@tf.keras.saving.register_keras_serializable(package="Scale16s", name="regression_loss_normal")
def regression_loss_normal(y_true, y_pred):
    true_mean = tf.reduce_mean(y_true)
    true_variance = tf.math.reduce_variance(y_true)

    y_pred_normalize = (y_pred - true_mean) / tf.sqrt(true_variance)

    return tf.reduce_sum(
        1.0 - tf.math.exp(-1.0*tf.square(y_pred_normalize) / 2.0)
    )