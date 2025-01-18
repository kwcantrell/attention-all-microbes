from __future__ import annotations

import tensorflow as tf

from aam.models.multihead_attention_with_linear_biases import MultiHeadAttention


@tf.keras.saving.register_keras_serializable(package="MultiHeadAttentionPooling")
class MultiHeadAttentionPooling(tf.keras.layers.Layer):
    def __init__(self, normalize_output, num_heads=4, use_residual_connections=True, use_linear_bias=False):
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.normalize_output = normalize_output
        self.use_residual_connections = use_residual_connections
        self.use_linear_bias = use_linear_bias

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        key_dim = int(hidden_dim // self.num_heads)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float32)
        if self.use_linear_bias:
            self.attention = MultiHeadAttention(
                self.num_heads,
                key_dim=key_dim,
                dropout=0.1,
            )

        else:
            self.attention = tf.keras.layers.MultiHeadAttention(
                self.num_heads,
                key_dim=key_dim,
                dropout=0.1,
            )
        if self.use_residual_connections:
            self._rezero = self.add_weight(
                name="rezero_alpha", initializer=tf.keras.initializers.Zeros(), trainable=True, dtype=tf.float32
            )

    def get_config(self):
        config = super(MultiHeadAttentionPooling, self).get_config()
        config.update(
            {
                "normalize_output": self.normalize_output,
                "num_heads": self.num_heads,
                "use_residual_connections": self.use_residual_connections,
                "use_linear_bias": self.use_linear_bias,
            }
        )
        return config

    def call(self, inputs, mask=None, training=False):
        attention_mask = mask
        if mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=self.compute_dtype)
            attention_mask = tf.matmul(attention_mask, attention_mask, transpose_b=True)

        # disable dropout
        attention_output = self.attention(inputs, inputs, attention_mask=attention_mask, training=False)

        if self.use_residual_connections:
            print("Pooler residual connection...")
            attention_output = inputs + self._rezero * attention_output

        if mask is not None:
            mask = tf.cast(mask, dtype=self.compute_dtype)
            seq_len = tf.reduce_sum(mask, axis=1)
            output = tf.reduce_sum(attention_output * mask, axis=1) / seq_len
        else:
            output = tf.reduce_mean(attention_output, axis=1)

        if self.normalize_output:
            print("Pooler Normalizing outputs...")
            output = self.norm(output)

            if self.compute_dtype == "float16":
                # output_tensor will always be float32
                # so we need to cast it back to float16
                output = tf.cast(output, dtype=tf.float16)
        print("Pooler exit...", self.trainable)
        return output
