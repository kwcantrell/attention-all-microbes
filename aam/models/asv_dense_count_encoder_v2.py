import tensorflow as tf

from aam.models.convolution_block import ConvolutionBlock
from aam.models.feedforward import FeedForward


@tf.keras.saving.register_keras_serializable(package="ASVDenseCountEncoderV2")
class ASVDenseCountEncoderV2(tf.keras.Model):
    def __init__(
        self,
        non_pool_blocks_per_layer=3,
        num_filters=16,
        kernel_size=3,
        pool_size=2,
        dropout_rate=0.0,
        max_pool=False,
        **kwargs,
    ):
        super(ASVDenseCountEncoderV2, self).__init__(**kwargs)
        self.non_pool_blocks_per_layer = non_pool_blocks_per_layer
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.max_pool = max_pool

    def build(self, input_shape):
        if self.built:
            print("ASVDenseCountEncoderV2 is already built")
            return

        asv_embeddings, dense_counts = input_shape

        dense_size = dense_counts[-1]
        conv_layers = []
        i = 0
        while dense_size > asv_embeddings[-1]:
            conv_layers += [tf.keras.layers.Reshape([-1, 1])]
            for _ in range(self.non_pool_blocks_per_layer):
                conv_layers += [
                    ConvolutionBlock(
                        self.num_filters,
                        self.kernel_size,
                        pool_size=0,
                        dropout_rate=self.dropout_rate,
                    )
                ]
            conv_layers += [
                ConvolutionBlock(
                    self.num_filters,
                    self.kernel_size,
                    pool_size=self.pool_size,
                    dropout_rate=self.dropout_rate,
                    use_max_pool=self.max_pool,
                )
            ]
            conv_layers += [
                tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1))
            ]

            dense_size /= self.pool_size
            i += 1
        print(f"{i} dense conv layers")

        self.dense_count_encoder = tf.keras.Sequential(
            conv_layers + [FeedForward(outdim=asv_embeddings[-1]), FeedForward()]
        )

        if self.dropout_rate > 0.0:
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

        super(ASVDenseCountEncoderV2, self).build(input_shape)

    def _log1p_relative_abundance(self, dense_counts):
        # compute relative abundance
        dense_counts = tf.cast(dense_counts, dtype=tf.float32)
        total_counts = tf.reduce_sum(dense_counts, axis=-1)
        depth = tf.reduce_max(total_counts)
        dense_counts /= depth

        # normalize counts
        dense_counts = tf.math.log1p(dense_counts)
        return dense_counts

    def call(self, inputs, training=False):
        training = training and self.trainable

        asv_embeddings, dense_counts = inputs
        dense_counts = self._log1p_relative_abundance(dense_counts)

        count_embeddings = self.dense_count_encoder(dense_counts, training=training)

        if self.dropout_rate > 0.0:
            count_embeddings = self.dropout(count_embeddings, training=training)

        # residual step
        output = asv_embeddings + self._rezero * count_embeddings

        return output

    def get_config(self):
        config = super(ASVDenseCountEncoderV2, self).get_config()
        config.update(
            {
                "non_pool_blocks_per_layer": self.non_pool_blocks_per_layer,
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
                "dropout_rate": self.dropout_rate,
                "max_pool": self.max_pool,
            }
        )
        return config
