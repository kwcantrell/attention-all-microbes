import tensorflow as tf
from aam.losses import _pairwise_distances, mae_loss


def pairwise_mae(batch_size):
    @tf.keras.saving.register_keras_serializable(
        package="amplicon_gpt.metrics")
    class PairwiseMAE(tf.keras.metrics.Metric):
        def __init__(self, name='pairwise_mae', dtype=tf.float32):
            super().__init__(name=name, dtype=dtype)
            self.loss = self.add_weight(name='rl',
                                        initializer='zero',
                                        dtype=tf.float32)
            self.i = self.add_weight(name='i',
                                     initializer='zero',
                                     dtype=tf.float32)

        def update_state(self, y_true, y_pred, sample_weight):
            pairwise_mae = tf.abs(_pairwise_distances(y_pred) - y_true)
            self.loss.assign_add(tf.reduce_sum(pairwise_mae))
            COMPARISONS = (batch_size * batch_size) - batch_size / 2.0
            self.i.assign_add(tf.constant(COMPARISONS, dtype=tf.float32))

        def result(self):
            return self.loss / self.i

    return PairwiseMAE()


class MAE(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self,
                 input_mean=None,
                 input_std=None,
                 name="mae",
                 dtype=None):
        super().__init__(fn=mae_loss(input_mean, input_std),
                         name=name,
                         dtype=dtype)
        self.input_mean = input_mean
        self.input_std = input_std
        self._direction = "down"

    def get_config(self):
        return {"input_mean": self.input_mean,
                "input_std": self.input_std,
                "name": self.name,
                "dtype": self.dtype}
