import tensorflow as tf


def denormalize(tensor, mean, std):
    return tensor*std + mean

def mae(batch_size, mean, std):
    
    @tf.keras.saving.register_keras_serializable(
        package="sepsis.metrics")
    class MAE(tf.keras.metrics.Metric):
        def __init__(self, name='mae', dtype=tf.float32):
            super().__init__(name=name, dtype=dtype)
            self.loss = self.add_weight(name='rl',
                                        initializer='zero',
                                        dtype=tf.float32)
            self.i = self.add_weight(name='i',
                                     initializer='zero',
                                     dtype=tf.float32)

        def update_state(self, y_true, y_pred, sample_weight):
            y_true = denormalize(y_true, mean, std)
            y_pred = denormalize(y_pred, mean, std)
            mae = tf.abs(y_true - y_pred)
            self.loss.assign_add(tf.reduce_sum(mae))
            self.i.assign_add(tf.constant(batch_size, dtype=tf.float32))

        def result(self):
            return self.loss / self.i

    return MAE()
