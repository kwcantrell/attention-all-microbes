import tensorflow as tf


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, pool_size=0, dropout_rate=0.0, **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate

        self.conv_inner = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding="same",
                ),
                tf.keras.layers.Activation("gelu"),
            ]
        )

        if self.pool_size > 0:
            self.conv_outer = tf.keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.pool_size,
                strides=self.pool_size,
                padding="same",
            )
            self.res_pool = tf.keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.pool_size,
                strides=self.pool_size,
                padding="same",
            )
        else:
            self.conv_outer = tf.keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=1,
                padding="same",
            )

        if dropout_rate > 0.0:
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, inputs, training=False):
        output = self.conv_inner(inputs)
        output = self.conv_outer(output)

        if self.dropout_rate > 0.0:
            output = self.dropout(output, training=training)

        # residual step
        if self.pool_size > 0:
            inputs = self.res_pool(inputs)
        output = inputs + self._rezero * output

        return output

    def get_config(self):
        config = super(ConvolutionBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
