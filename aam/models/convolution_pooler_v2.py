import tensorflow as tf


class PoolingBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(PoolingBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        if isinstance(input_shape, tuple):
            input_shape, _, _, _ = input_shape
        if len(input_shape) != 3:
            raise Exception("Must be rank 3!")

        self.pool = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=1,
                    kernel_size=self.kernel_size,
                    strides=self.kernel_size,
                    padding="same",
                ),
                tf.keras.layers.Conv1D(
                    filters=self.filters, kernel_size=1, strides=1, padding="same"
                ),
            ],
            name="pool",
        )

        self.conv_blocks = tf.keras.Sequential(
            [
                NonPoolingBlock(filters=self.filters, kernel_size=3, strides=1),
                NonPoolingBlock(filters=self.filters, kernel_size=3, strides=1),
            ],
            name="conv_blocks",
        )

        self.res_pool = tf.keras.layers.MaxPool1D(
            pool_size=self.kernel_size,
            strides=self.kernel_size,
            padding="same",
            name="res_pool",
        )

    def call(self, inputs, training=False):
        dense_counts, modified_counts, embeddings, shifted_mask = inputs

        # pool non modified counts for reconstruction later
        pooled_counts = self.pool(dense_counts)

        pooled_mod_counts = self.pool(modified_counts)
        pooled_embeddings = pooled_mod_counts + embeddings
        output = self.conv_blocks(pooled_embeddings)

        # used to identify which blocks were altered
        modified_blocks = self.res_pool(shifted_mask)
        modified_blocks = tf.squeeze(modified_blocks, axis=-1)

        return pooled_counts, output, modified_blocks

    def get_config(self):
        config = super(PoolingBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
            }
        )
        return config


class NonPoolingBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(NonPoolingBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise Exception("Must be rank 3!")

        self.conv_block = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=1,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding="same",
                ),
                tf.keras.layers.Conv1D(
                    filters=self.filters,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                ),
                tf.keras.layers.Activation("gelu"),
            ]
        )
        self._rezero = self.add_weight(
            name="rezero",
            dtype=tf.float32,
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, inputs, training=False):
        output = self.conv_block(inputs)
        output = inputs + tf.cast(self._rezero, dtype=self.compute_dtype) * output
        return output

    def get_config(self):
        config = super(NonPoolingBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
            }
        )
        return config


@tf.keras.saving.register_keras_serializable(package="ConvolutionPoolerV2")
class ConvolutionPoolerV2(tf.keras.layers.Layer):
    def __init__(self, filters=8, kernel_size=3, dropout_rate=0.0, **kwargs):
        super(ConvolutionPoolerV2, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        if len(input_shape) == 2:
            cur_dim = input_shape[-1]
            pool_layers = [tf.keras.layers.Reshape([-1, 1])]
        else:
            cur_dim = input_shape[1]
            pool_layers = []

        i = 0
        while cur_dim > self.filters:
            pool_layers += [
                PoolingBlock(
                    self.filters,
                    self.kernel_size,
                ),
            ]
            cur_dim /= self.kernel_size
            i += 1
        print(f"{i} total pooling layers")
        self.pooler = tf.keras.Sequential(
            pool_layers + [tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))],
            name="pooler",
        )

    def call(self, inputs, training=False):
        return self.pooler(inputs)

    def get_config(self):
        return (
            super(ConvolutionPoolerV2, self)
            .get_config()
            .update(
                {
                    "filters": self.filters,
                    "kernel_size": self.kernel_size,
                    "dropout_rate": self.dropout_rate,
                }
            )
        )
