import tensorflow as tf

from aam.losses import _pairwise_distances, mae_loss


class PairwiseMAE(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = self.add_weight(name="rl", initializer="zero", dtype=tf.float32)
        self.i = self.add_weight(
            name="i",
            initializer="zero",
            dtype=tf.float32,
        )

        @tf.function
        def compute_compare(y_true, y_pred):
            batch_size = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
            return (batch_size * batch_size) - batch_size / 2.0

        self.compare = compute_compare

    def update_state(self, y_true, y_pred, sample_weight=None):
        pairwise_mae = tf.abs(_pairwise_distances(y_pred) - y_true)
        self.loss.assign_add(tf.reduce_sum(pairwise_mae))
        self.i.assign_add(self.compare(y_true, y_pred))

    def result(self):
        return self.loss / self.i

    def get_config(self):
        return super().get_config()


@tf.keras.saving.register_keras_serializable(package="MAE")
class MAE(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, shift=None, scale=None, dtype=None, **kwargs):
        super().__init__(fn=mae_loss(shift, scale), dtype=dtype, **kwargs)
        self.shift = shift
        self.scale = scale
        self._direction = "down"

    def get_config(self):
        base_config = super().get_config()
        config = {
            "shift": self.shift,
            "scale": self.scale,
            "name": self.name,
            "dtype": self.dtype,
        }
        return {**base_config, **config}
