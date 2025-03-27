import tensorflow as tf

from aam.models.conv_feedforward_v2 import ConvFeedForwardV2
from aam.models.convolution_pooler import ConvolutionPooler


@tf.keras.saving.register_keras_serializable(package="ASVDenseCountEncoderV3")
class ASVDenseCountEncoderV3(tf.keras.Model):
    def __init__(
        self,
        pool_conv_blocks_per_layer=2,
        ff_conv_blocks_per_layer=2,
        filters=32,
        kernel_size=3,
        pool_size=2,
        dropout_rate=0.0,
        max_pool=False,
        **kwargs,
    ):
        super(ASVDenseCountEncoderV3, self).__init__(**kwargs)
        self.ff_conv_blocks_per_layer = ff_conv_blocks_per_layer
        self.pool_conv_blocks_per_layer = pool_conv_blocks_per_layer
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.max_pool = max_pool

    def build(self, input_shape):
        if self.built:
            print("ASVDenseCountEncoderV3 is already built")
            return

        asv_embeddings, dense_counts = input_shape

        dense_size = dense_counts[-1]
        conv_layers = []
        i = 0
        while dense_size > asv_embeddings[-1]:
            conv_layers += [
                ConvolutionPooler(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    conv_blocks=self.pool_conv_blocks_per_layer,
                )
            ]
            dense_size /= 2
            i += 1
        print(f"{i} dense conv layers")

        self.dense_count_encoder = tf.keras.Sequential(
            conv_layers
            + [
                ConvFeedForwardV2(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    conv_blocks=self.ff_conv_blocks_per_layer,
                    outdim=asv_embeddings[-1],
                )
            ]
        )

        if self.dropout_rate > 0.0:
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

        super(ASVDenseCountEncoderV3, self).build(input_shape)

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
        config = super(ASVDenseCountEncoderV3, self).get_config()
        config.update(
            {
                "ff_conv_blocks_per_layer": self.ff_conv_blocks_per_layer,
                "pool_conv_blocks_per_layer": self.pool_conv_blocks_per_layer,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
                "dropout_rate": self.dropout_rate,
                "max_pool": self.max_pool,
            }
        )
        return config
