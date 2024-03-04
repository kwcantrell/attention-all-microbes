import tensorflow as tf
from amplicon_gpt.losses import _pairwise_distances


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
