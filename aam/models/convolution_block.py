import tensorflow as tf


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        num_blocks=8,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

        conv_blocks = []
        for _ in range(self.num_blocks):
            conv_blocks += [
                tf.keras.layers.Conv1D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding="same",
                ),
                tf.keras.layers.Activation("gelu"),
            ]
        self.conv_blocks = tf.keras.Sequential(conv_blocks, name="conv_blocks")

        if dropout_rate > 0.0:
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, inputs, training=False):
        output = self.conv_blocks(inputs)

        if self.dropout_rate > 0.0:
            output = self.dropout(output, training=training)

        if self.filters == 1:
            inputs = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        output = inputs + self._rezero * output

        return output

    def get_config(self):
        config = super(ConvolutionBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "conv_blocks": self.conv_blocks,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
