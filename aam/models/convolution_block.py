import tensorflow as tf


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        pool=False,
        num_blocks=1,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool = pool
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise Exception("Must be rank 3!")

        conv_blocks = []
        for _ in range(self.num_blocks):
            conv_blocks += [
                tf.keras.layers.Conv1D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding="same",
                ),
                tf.keras.layers.Activation("gelu"),
            ]

        if self.pool:
            conv_blocks += [
                tf.keras.layers.Conv1D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=self.kernel_size,
                    padding="same",
                ),
                tf.keras.layers.Activation("gelu"),
            ]
            self.res_pool = tf.keras.layers.MaxPool1D(
                pool_size=self.kernel_size, strides=self.kernel_size, padding="same"
            )
        self.conv_blocks = tf.keras.Sequential(conv_blocks, name="conv_blocks")
        if self.dropout_rate > 0.0:
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        # self._rezero = self.add_weight(
        #     name="ff_rezero_alpha",
        #     initializer=tf.keras.initializers.Zeros(),
        #     trainable=True,
        #     dtype=tf.float32,
        # )

    def call(self, inputs, training=False):
        output = self.conv_blocks(inputs)

        if self.dropout_rate > 0.0:
            output = self.dropout(output, training=training)

        if self.filters == 1:
            inputs = tf.reduce_mean(inputs, axis=-1, keepdims=True)

        if self.pool:
            inputs = self.res_pool(inputs)
        output = inputs + output

        if self.dropout_rate > 0.0:
            output = self.dropout(output, training=training)
        return output

    def get_config(self):
        config = super(ConvolutionBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "pool": self.pool,
                "conv_blocks": self.conv_blocks,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
