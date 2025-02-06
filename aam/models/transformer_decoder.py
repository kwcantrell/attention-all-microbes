from __future__ import annotations

import tensorflow as tf
import tensorflow_models as tfm

from aam.models.linear_attention_bias import LinearBiasSoftmax


@tf.keras.saving.register_keras_serializable(package="TransformerDecoder")
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        activation="gelu",
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        norm_epsilon=1e-6,
        normalize_outputs=False,
        use_residual_connections=True,
        use_linear_bias=True,
        **kwargs,
    ):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._attention_dropout_rate = attention_dropout_rate
        self._norm_epsilon = norm_epsilon
        self.normalize_outputs = normalize_outputs
        self.use_residual_connections = use_residual_connections
        self.use_linear_bias = use_linear_bias

    def build(self, input_shape):
        asv_input_shape, gotu_input_shape = input_shape
        self.hidden_dim = asv_input_shape[-1]
        """Implements build() for the layer."""
        self.causal_encoders = []
        self.cross_attention_encoders = []

        def get_transformer(i, name):
            transformer = tfm.nlp.layers.ReZeroTransformer(
                num_attention_heads=self.num_attention_heads,
                inner_dim=self._intermediate_size,
                inner_activation=self._activation,
                dropout_rate=self._dropout_rate,
                attention_dropout_rate=0.0,
                share_rezero=True,
                name=name,
            )
            linear_bias_softmax = LinearBiasSoftmax()
            if "causal_encoder" in name:
                transformer.build(gotu_input_shape)
                transformer._attention_layer._build_from_signature(
                    gotu_input_shape, gotu_input_shape
                )
            else:
                transformer.build(asv_input_shape)
                transformer._attention_layer._build_from_signature(
                    gotu_input_shape, asv_input_shape
                )

            if self.use_linear_bias:
                setattr(transformer._attention_layer, "_softmax", linear_bias_softmax)
            return transformer

        for i in range(self.num_layers):
            self.causal_encoders.append(get_transformer(i, ("causal_encoder_%d" % i)))

            self.cross_attention_encoders.append(
                get_transformer(i, ("cross_attn_encoder_%d" % i))
            )
        if self.normalize_outputs:
            self.output_normalization = tf.keras.layers.LayerNormalization(
                epsilon=1e-6, dtype=tf.float32
            )
        super(TransformerDecoder, self).build(input_shape)

    def get_config(self):
        config = {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self._intermediate_size,
            "activation": self._activation,
            "dropout_rate": self._dropout_rate,
            "attention_dropout_rate": self._attention_dropout_rate,
            "norm_epsilon": self._norm_epsilon,
            "normalize_outputs": self.normalize_outputs,
            "use_residual_connections": self.use_residual_connections,
            "use_linear_bias": self.use_linear_bias,
        }
        base_config = super(TransformerDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, asv_mask=None, gotu_mask=None, training=False):
        """Return the output of the encoder.

        Args:
          causal_inputs: A tensor with shape `(batch_size, input_length,
            hidden_size)`.
          asv_mask: A mask for the encoder self-attention layer with shape
            `(batch_size, asv_input_length, 1)`.
          gotu_mask: A mask for the encoder self-attention layer with shape
            `(batch_size, gotu_input_length, 1)`.

        Returns:
          Output of encoder which is a `float32` or `float16` tensor with shape
            `(batch_size, input_length, hidden_size)`.
        """
        asv_inputs, gotu_inputs = inputs
        causal_inputs = gotu_inputs
        gotu_shape = tf.shape(causal_inputs)
        batch_dim = gotu_shape[0]
        g_seq_len = gotu_shape[1]
        causal_mask = tf.linalg.band_part(
            tf.ones([batch_dim, g_seq_len, g_seq_len], dtype=self.compute_dtype), -1, 0
        )
        if gotu_mask is not None:
            causal_mask = causal_mask * tf.matmul(
                gotu_mask, gotu_mask, transpose_b=True
            )
        for layer_idx in range(self.num_layers):
            causal_inputs = self.causal_encoders[layer_idx](
                [causal_inputs, causal_mask], training=training
            )

        query_inputs = causal_inputs
        key_inputs = asv_inputs
        # attention_mask = None
        # if asv_mask is not None and gotu_mask is not None:
        attention_mask = tf.matmul(gotu_mask, asv_mask, transpose_b=True)
        for layer_idx in range(self.num_layers):
            query_inputs = self.cross_attention_encoders[layer_idx](
                [query_inputs, key_inputs, attention_mask], training=training
            )
        output_tensor = query_inputs

        if self.normalize_outputs:
            print("Encoder normalizing outputs...")
            output_tensor = self.output_normalization(output_tensor)

        if self.compute_dtype == "float16":
            # output_tensor will always be float32
            # so we need to cast it back to float16
            output_tensor = tf.cast(output_tensor, dtype=tf.float16)
        print("Decoder exit...", self.trainable)
        return output_tensor
