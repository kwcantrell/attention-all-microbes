from __future__ import annotations

import tensorflow as tf
import tensorflow_models as tfm


@tf.keras.saving.register_keras_serializable(package="TransformerDecoder")
class TransformerDecoder(tf.keras.layers.Layer):
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
        use_layer_norm=False,
        share_rezero=True,
        **kwargs,
    ):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._attention_dropout_rate = attention_dropout_rate
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self.use_layer_norm = use_layer_norm

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        """Implements build() for the layer."""
        self.encoder_layers = []
        self.decoder_layers = []
        for i in range(self.num_layers):
            self.encoder_layers.append(
                tfm.nlp.layers.ReZeroTransformer(
                    num_attention_heads=self.num_attention_heads,
                    inner_dim=self._intermediate_size,
                    inner_activation=self._activation,
                    dropout_rate=self._dropout_rate,
                    attention_dropout_rate=self._dropout_rate,
                    share_rezero=True,
                    name=("layer_%d" % i),
                )
            )

            self.decoder_layers.append(
                tfm.nlp.layers.ReZeroTransformer(
                    num_attention_heads=self.num_attention_heads,
                    inner_dim=self._intermediate_size,
                    inner_activation=self._activation,
                    dropout_rate=self._dropout_rate,
                    attention_dropout_rate=self._dropout_rate,
                    share_rezero=True,
                    name=("layer_%d" % i),
                )
            )
        if self.use_layer_norm:
            self.output_normalization = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float32)
        super(TransformerDecoder, self).build(input_shape)

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
            "use_layer_norm": self.use_layer_norm,
        }
        base_config = super(TransformerDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, asv_inputs, gotu_inputs, asv_mask=None, gotu_mask=None, training=False):
        """Return the output of the encoder.

        Args:
          encoder_inputs: A tensor with shape `(batch_size, input_length,
            hidden_size)`.
          asv_mask: A mask for the encoder self-attention layer with shape
            `(batch_size, asv_input_length, 1)`.
          gotu_mask: A mask for the encoder self-attention layer with shape
            `(batch_size, gotu_input_length, 1)`.

        Returns:
          Output of encoder which is a `float32` or `float16` tensor with shape
            `(batch_size, input_length, hidden_size)`.
        """

        encoder_inputs = gotu_inputs
        gotu_shape = tf.shape(encoder_inputs)
        batch_dim = gotu_shape[0]
        g_seq_len = gotu_shape[1]
        causal_mask = tf.linalg.band_part(tf.ones([batch_dim, g_seq_len, g_seq_len], dtype=self.compute_dtype), -1, 0)
        if gotu_mask is not None:
            causal_mask = causal_mask * tf.matmul(gotu_mask, gotu_mask, transpose_b=True)
        for layer_idx in range(self.num_layers):
            encoder_inputs = self.encoder_layers[layer_idx]([encoder_inputs, causal_mask], training=training)

        decoder_inputs = encoder_inputs
        attention_mask = None
        if asv_mask is not None and gotu_mask is not None:
            attention_mask = tf.matmul(gotu_mask, asv_mask, transpose_b=True)
        for layer_idx in range(self.num_layers):
            decoder_inputs = self.decoder_layers[layer_idx]([decoder_inputs, asv_inputs, attention_mask], training=training)
        output_tensor = decoder_inputs
        if self.use_layer_norm:
            output_tensor = self.output_normalization(output_tensor)
        if self.compute_dtype == "float16":
            # output_tensor will always be float32
            # so we need to cast it back to float16
            output_tensor = tf.cast(output_tensor, dtype=tf.float16)
        return output_tensor
