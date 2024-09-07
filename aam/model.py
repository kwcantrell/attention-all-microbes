import tensorflow as tf

@tf.keras.saving.register_keras_serializable(package="AAM")
class AAM(tf.keras.Model):
    def __init__(self, model, mean, std):
        self.mean = mean
        self.std = std