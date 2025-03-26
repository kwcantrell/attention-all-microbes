import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="ConvolutionPooler")
class ConvolutionPooler(tf.keras.layers.Layer):
    def __init__(self, filters=32, kernel_size=3, conv_blocks=3, **kwargs):
        super(ConvolutionPooler, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_blocks = conv_blocks

        conv_layers = [tf.keras.layers.Reshape([-1, 1])]
        for _ in range(self.conv_blocks):
            conv_layers += [
                tf.keras.layers.Conv1D(
                    filters=self.filters,
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
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding="same",
                ),
                tf.keras.layers.Conv1D(
                    filters=self.filters,
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
        conv_output = self.conv_layers(inputs)
        pooler_input = inputs + self._rezero * conv_output

        pooler_output = self.pooler(pooler_input)
        pooler_input = self.res_pool(pooler_input)
        output = pooler_input + self._rezero * pooler_output
        return output

    def get_config(self):
        return (
            super(ConvolutionPooler, self)
            .get_config()
            .update(
                {
                    "filters": self.filters,
                    "kernel_size": self.kernel_size,
                    "conv_blocks": self.conv_blocks,
                }
            )
        )
