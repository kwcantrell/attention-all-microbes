import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="ConvolutionPooler")
class ConvolutionPooler(tf.keras.layers.Layer):
    def __init__(self, num_filters=32, kernel_size=3, num_layers=3, **kwargs):
        super(ConvolutionPooler, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers

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

        self.pooler = tf.keras.Sequential(
            [
                tf.keras.layers.Reshape([-1, 1]),
                tf.keras.layers.Conv1D(
                    filters=self.num_filters,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding="same",
                ),
                tf.keras.layers.Conv1D(
                    filters=self.num_filters,
                    kernel_size=2,
                    strides=2,
                    padding="same",
                ),
                tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1)),
            ]
        )
        self.res_pool = tf.keras.Sequential(
            [
                tf.keras.layers.Reshape([-1, 1]),
                tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding="same"),
                tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1)),
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
        output = self.res_pool(conv_outputs) + self._rezero * self.pooler(conv_outputs)
        return output

    def get_config(self):
        return (
            super(ConvolutionPooler, self)
            .get_config()
            .update(
                {
                    "num_filters": self.num_filters,
                    "kernel_size": self.kernel_size,
                    "num_layers": self.num_layers,
                }
            )
        )
