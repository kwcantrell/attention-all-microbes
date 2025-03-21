import tensorflow as tf

from aam.models.convolution_block import ConvolutionBlock


@tf.keras.saving.register_keras_serializable(package="ASVDenseCountEncoder")
class ASVDenseCountEncoder(tf.keras.Model):
    def __init__(self, num_filters=16, kernel_size=3, pool_size=2, **kwargs):
        super(ASVDenseCountEncoder, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size

    def build(self, input_shape):
        if self.built:
            print("ASVDenseCountEncoder is already built")
            return

        asv_embeddings, dense_counts = input_shape

        dense_size = dense_counts[-1]
        conv_layers = [tf.keras.layers.Input([dense_size, 1])]
        i = 0
        while dense_size > 512:
            conv_layers += [
                ConvolutionBlock(self.num_filters, self.kernel_size, pool_size=0),
                ConvolutionBlock(self.num_filters, self.kernel_size, pool_size=0),
                ConvolutionBlock(self.num_filters, self.kernel_size, pool_size=0),
                ConvolutionBlock(
                    self.num_filters, self.kernel_size, pool_size=self.pool_size
                ),
            ]
            dense_size /= self.pool_size
            i += 1
        print(f"{i} dense conv layers")

        self.dense_count_encoder = tf.keras.Sequential(
            conv_layers
            + [
                tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1)),
                tf.keras.layers.Dense(asv_embeddings[-1]),
            ]
        )
        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )
        super(ASVDenseCountEncoder, self).build(input_shape)

    def call(
        self,
        inputs,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable

        asv_embeddings, dense_counts = inputs
        count_embeddings = self.dense_count_encoder(dense_counts, training=training)

        # residual step
        encoder_input = asv_embeddings + self._rezero * count_embeddings
        return encoder_input

    def get_config(self):
        return (
            super()
            .get_config()
            .update(
                {
                    "num_filters": self.num_filters,
                    "kernel_size": self.kernel_size,
                    "pool_size": self.pool_size,
                }
            )
        )
