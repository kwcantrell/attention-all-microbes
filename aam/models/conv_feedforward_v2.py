import tensorflow as tf

from aam.models.convolution_block import ConvolutionBlock


@tf.keras.saving.register_keras_serializable(package="ConvFeedForwardV2")
class ConvFeedForwardV2(tf.keras.layers.Layer):
    def __init__(
        self,
        filters=32,
        kernel_size=3,
        conv_blocks=8,
        pool=0,
        outdim=None,
        conv_dropout_rate=0.0,
        ff_dropout_rate=0.0,
        **kwargs,
    ):
        super(ConvFeedForwardV2, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_blocks = conv_blocks
        self.pool = pool
        self.outdim = outdim
        self.conv_dropout_rate = conv_dropout_rate
        self.ff_dropout_rate = ff_dropout_rate

    def build(self, input_shape):
        units = input_shape[-1]

        if len(input_shape) == 2:
            conv_layers = [tf.keras.layers.Reshape([-1, 1])]
        else:
            conv_layers = []
        for _ in range(self.conv_blocks):
            conv_layers += [ConvolutionBlock(self.filters, self.kernel_size)]
        conv_layers += [
            tf.keras.layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=-1, keepdims=len(input_shape) != 2)
            )
        ]
        self.conv_layers = tf.keras.Sequential(conv_layers, name="conv_layers")

        if self.conv_dropout_rate > 0.0:
            self.conv_dropout = tf.keras.layers.Dropout(self.conv_dropout_rate)

        if self.pool < 0:
            self.ff = tf.keras.layers.Dense(units // 2, activation="gelu")
            self.res_pool = tf.keras.layers.Dense(units // 2)
        elif self.pool > 0:
            self.ff = tf.keras.layers.Dense(units * 2, activation="gelu")
            self.res_pool = tf.keras.layers.Dense(units * 2)
        elif self.outdim is not None:
            self.ff = tf.keras.layers.Dense(self.outdim, activation="gelu")
            self.res_pool = tf.keras.layers.Dense(self.outdim)
        else:
            self.ff = tf.keras.layers.Dense(units, activation="gelu")
        if self.ff_dropout_rate > 0.0:
            self.ff_dropout = tf.keras.layers.Dropout(self.ff_dropout_rate)

        self._conv_rezero = self.add_weight(
            name="conv_rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )
        self._ff_rezero = self.add_weight(
            name="ff_rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, inputs, training=False):
        conv_output = self.conv_layers(inputs)
        if self.conv_dropout_rate > 0.0:
            conv_output = self.conv_dropout(conv_output, training=training)
        ff_input = inputs + self._conv_rezero * conv_output

        ff_output = self.ff(ff_input)
        if self.ff_dropout_rate > 0.0:
            ff_output = self.ff_dropout(ff_output, training=training)
        if self.pool or self.outdim is not None:
            ff_input = self.res_pool(ff_input)
        output = ff_input + self._ff_rezero * ff_output
        return output

    def get_config(self):
        return (
            super(ConvFeedForwardV2, self)
            .get_config()
            .update(
                {
                    "filters": self.filters,
                    "kernel_size": self.kernel_size,
                    "conv_blocks": self.conv_blocks,
                    "pool": self.pool,
                    "outdim": self.outdim,
                    "conv_dropout_rate": self.conv_dropout_rate,
                    "ff_dropout_rate": self.ff_dropout_rate,
                }
            )
        )
