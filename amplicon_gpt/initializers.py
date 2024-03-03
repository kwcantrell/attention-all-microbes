import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="amplicon_gpt",
                                             name="MemoryHead")
class UnitUniform(tf.keras.initializers.Initializer):
    def __init__(self, fan_out):
        super().__init__()
        self.fan_out = fan_out

    def __call__(self, shape, dtype=None):
        return tf.ones(shape=shape, dtype=tf.float32) / self.fan_out

    def get_config(self):
        return {
            "fan_out": self.fan_out
        }
