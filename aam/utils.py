import tensorflow as tf


def float_mask(tensor: tf.Tensor, dtype=tf.float32):
    """creates a mask for nonzero elements of tensor. I.e. mask*tensor = tensor and
    (1. - mask) * tensor = 0

    Args:
        tensor (tf.Tensor): a tensor of type float

    Returns:
        tf.Tensor:
    """
    mask = tf.cast(tf.not_equal(tensor, 0), dtype=dtype)
    return mask


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate=0.001, warmup=4000, d_model=128):
        self.initial_learning_rate = initial_learning_rate
        self.warmup = tf.cast(warmup, dtype=tf.float32)
        self.d_model = tf.cast(d_model, dtype=tf.float32)

    def __call__(self, step):
        s = tf.cast(step, dtype=tf.float32)
        l1 = tf.pow(s, -0.5)
        l2 = s * tf.pow(self.warmup, -1.5)
        return tf.pow(self.d_model, -0.5) * tf.where(l1 < l2, l1, l2)

    def get_config(self):
        config = {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup": self.warmup,
        }
        return {**config}


class LRDecrease(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr=0.0005, decay=0.99999):
        self.lr = tf.Variable(lr, trainable=False)
        self.decay = 0.99995

    def __call__(self, step):
        lr = self.lr * self.decay
        lr = tf.math.maximum(0.0001, lr)
        self.lr.assign(lr)

        return lr

    def get_config(self):
        config = {}
        return {**config}
