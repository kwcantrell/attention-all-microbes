import tensorflow as tf


def float_mask(tensor: tf.Tensor):
    """creates a mask for nonzero elements of tensor. I.e. mask*tensor = tensor and
    (1. - mask) * tensor = 0

    Args:
        tensor (tf.Tensor): a tensor of type float

    Returns:
        tf.Tensor:
    """
    mask = tf.cast(tf.not_equal(tensor, 0), dtype=tf.float32)
    return mask
