import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="ConvFeedForward")
class ConvFeedForward(tf.keras.layers.Layer):
    def __init__(
        self,
        num_filters=32,
        kernel_size=3,
        num_layers=8,
        pool=0,
        outdim=None,
        **kwargs,
    ):
        super(ConvFeedForward, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.pool = pool
        self.outdim = outdim

    def build(self, input_shape):
        units = input_shape[-1]

        conv_layers = [tf.keras.layers.Reshape([-1, 1])]
        for _ in range(self.num_layers):
            conv_layers += [
                tf.keras.layers.Conv1D(
                    filters=self.num_filters,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding="same",
                )
            ]
        self.conv_layers = tf.keras.Sequential(
            conv_layers
            + [tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1))],
            name="conv_layers",
        )

        if self.pool < 0:
            self.ff = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(units, activation="relu"),
                    tf.keras.layers.Dense(units // 2),
                ]
            )
            self.res_pool = tf.keras.layers.Dense(units // 2)
        elif self.pool > 0:
            self.ff = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(units, activation="relu"),
                    tf.keras.layers.Dense(units * 2),
                ]
            )
            self.res_pool = tf.keras.layers.Dense(units * 2)
        elif self.outdim is not None:
            self.ff = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(units, activation="relu"),
                    tf.keras.layers.Dense(self.outdim),
                ]
            )
            self.res_pool = tf.keras.layers.Dense(self.outdim)
        else:
            self.ff = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(units, activation="relu"),
                    tf.keras.layers.Dense(units),
                ]
            )
        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, inputs, training=False):
        conv_outputs = inputs + self._rezero * self.conv_layers(inputs)

        if self.pool or self.outdim is not None:
            conv_outputs = self.res_pool(conv_outputs)
        output = conv_outputs + self._rezero * self.ff(conv_outputs)
        return output

    def get_config(self):
        return (
            super(ConvFeedForward, self)
            .get_config()
            .update(
                {
                    "num_filters": self.num_filters,
                    "kernel_size": self.kernel_size,
                    "num_layers": self.num_layers,
                    "pool": self.pool,
                    "outdim": self.outdim,
                }
            )
        )
