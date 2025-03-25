import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="FeedForward")
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, pool=0, outdim=None, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.pool = pool
        self.outdim = outdim

    def build(self, input_shape):
        units = input_shape[-1]

        if self.pool < 0:
            self.dense = tf.keras.layers.Dense(units // 2)
            self.res_pool = tf.keras.layers.Dense(units // 2)
        elif self.pool > 0:
            self.dense = tf.keras.layers.Dense(units * 2)
            self.res_pool = tf.keras.layers.Dense(units * 2)
        elif self.outdim is not None:
            self.dense = tf.keras.layers.Dense(self.outdim)
            self.res_pool = tf.keras.layers.Dense(self.outdim)
        else:
            self.dense = tf.keras.layers.Dense(units)

        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, inputs, training=False):
        output = self.dense(inputs)

        # residual step
        if self.pool or self.outdim is not None:
            inputs = self.res_pool(inputs)
        output = inputs + self._rezero * output
        return output

    def get_config(self):
        return (
            super(FeedForward, self)
            .get_config()
            .update({"pool": self.pool, "outdim": self.outdim})
        )
