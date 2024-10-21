import tensorflow as tf
import tensorflow_models as tfm


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        activation="gelu",
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        **kwargs,
    ):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._attention_dropout_rate = attention_dropout_rate
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        """Implements build() for the layer."""
        self.encoder_layers = []
        for i in range(self.num_layers):
            self.encoder_layers.append(
                tfm.nlp.layers.ReZeroTransformer(
                    num_attention_heads=self.num_attention_heads,
                    inner_dim=self._intermediate_size,
                    inner_activation=self._activation,
                    dropout_rate=self._dropout_rate,
                    attention_dropout_rate=self._dropout_rate,
                    name=("layer_%d" % i),
                )
            )
        self.output_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        super(TransformerEncoder, self).build(input_shape)

    def get_config(self):
        config = {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self._intermediate_size,
            "activation": self._activation,
            "dropout_rate": self._dropout_rate,
            "attention_dropout_rate": self._attention_dropout_rate,
            "use_bias": self._use_bias,
            "norm_first": self._norm_first,
            "norm_epsilon": self._norm_epsilon,
        }
        base_config = super(TransformerEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, encoder_inputs, mask=None, training=False):
        """Return the output of the encoder.

        Args:
          encoder_inputs: A tensor with shape `(batch_size, input_length,
            hidden_size)`.
          attention_mask: A mask for the encoder self-attention layer with shape
            `(batch_size, input_length, input_length)`.

        Returns:
          Output of encoder which is a `float32` tensor with shape
            `(batch_size, input_length, hidden_size)`.
        """
        attention_mask = mask
        if attention_mask is not None:
            attention_mask = tf.matmul(attention_mask, attention_mask, transpose_b=True)

        for layer_idx in range(self.num_layers):
            encoder_inputs = self.encoder_layers[layer_idx](
                [encoder_inputs, attention_mask], training=training
            )
        output_tensor = encoder_inputs
        return output_tensor
