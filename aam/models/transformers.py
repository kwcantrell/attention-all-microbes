from __future__ import annotations

import tensorflow as tf
import tensorflow_models as tfm

from aam.models.linear_attention_bias import LinearBiasSoftmax


@tf.keras.saving.register_keras_serializable(package="TransformerEncoder")
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
        use_residual_connections=False,
        use_linear_bias=True,
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
        self.use_residual_connections = use_residual_connections
        self.use_linear_bias = use_linear_bias

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            shape = input_shape[0]
        else:
            shape = input_shape
        self.hidden_dim = shape[-1]
        if self.use_residual_connections:
            self._rezero = self.add_weight(
                name="rezero_alpha",
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )
        linear_bias_softmax = LinearBiasSoftmax()
        print("Using linear bias")

        def get_transformer(i):
            transformer = tfm.nlp.layers.ReZeroTransformer(
                num_attention_heads=self.num_attention_heads,
                inner_dim=self._intermediate_size,
                inner_activation=self._activation,
                dropout_rate=self._dropout_rate,
                attention_dropout_rate=0.0,
                share_rezero=True,
                name=("layer_%d" % i),
            )
            transformer.build(shape)
            transformer._attention_layer._build_from_signature(shape, shape)

            if self.use_linear_bias:
                setattr(transformer._attention_layer, "_softmax", linear_bias_softmax)
            return transformer

        self.encoder_layers = []
        for i in range(self.num_layers):
            self.encoder_layers.append(get_transformer(i))
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
            "use_residual_connections": self.use_residual_connections,
            "use_linear_bias": self.use_linear_bias,
        }
        base_config = super(TransformerEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, mask=None, training=False):
        """Return the output of the encoder.

        Args:
          inputs: A tensor with shape `(batch_size, input_length,
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

        if isinstance(inputs, (list, tuple)):
            output_tensor, key_value = inputs
            inputs = output_tensor
            for layer_idx in range(self.num_layers):
                output_tensor = tf.cast(
                    self.encoder_layers[layer_idx](
                        [output_tensor, key_value, attention_mask], training=training
                    ),
                    dtype=self.compute_dtype,
                )
        else:
            output_tensor = inputs
            for layer_idx in range(self.num_layers):
                output_tensor = tf.cast(
                    self.encoder_layers[layer_idx](
                        [output_tensor, attention_mask], training=training
                    ),
                    dtype=self.compute_dtype,
                )

        if self.use_residual_connections:
            print("Encoder residual connection...")
            output_tensor = inputs + self._rezero * output_tensor

        if self.compute_dtype == "float16":
            # output_tensor will always be float32
            # so we need to cast it back to float16
            output_tensor = tf.cast(output_tensor, dtype=tf.float16)
        print("Encoder exit...", self.trainable)
        return output_tensor
