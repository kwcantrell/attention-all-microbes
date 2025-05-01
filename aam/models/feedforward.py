import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="FeedForward")
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        units = input_shape[-1]
        self.ff = tf.keras.layers.Dense(units, activation="gelu")
        self._rezero = self.add_weight(
            name="rezero",
            dtype=tf.float32,
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.built = True

    def call(self, inputs, training=False):
        output = self.ff(inputs)
        output = inputs + self._rezero * output
        return output
