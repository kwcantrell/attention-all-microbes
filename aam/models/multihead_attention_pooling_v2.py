from __future__ import annotations

import tensorflow as tf

from aam.models.linear_attention_bias import LinearBiasSoftmax


@tf.keras.saving.register_keras_serializable(package="MultiHeadAttentionPoolingV2")
class MultiHeadAttentionPoolingV2(tf.keras.layers.Layer):
    def __init__(
        self,
        normalize_output,
        num_heads=4,
        use_residual_connections=True,
        use_linear_bias=False,
        **kwargs,
    ):
        super(MultiHeadAttentionPoolingV2, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.normalize_output = normalize_output
        self.use_residual_connections = use_residual_connections
        self.use_linear_bias = use_linear_bias

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        key_dim = int(self.hidden_dim // self.num_heads)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float32)

        self.query = self.add_weight(
            "query",
            shape=[1, 1, self.hidden_dim],
            dtype=tf.float32,
            initializer="glorot_uniform",
            trainable=True,
        )
        self.attention = tf.keras.layers.MultiHeadAttention(
            self.num_heads,
            key_dim=key_dim,
            dropout=0.0,
        )
        self.attention._build_from_signature(
            [input_shape[0], 1, self.hidden_dim], input_shape
        )

        if self.use_linear_bias:
            print("Using linear bias")
            setattr(self.attention, "_softmax", LinearBiasSoftmax())

        if self.use_residual_connections:
            self._rezero = self.add_weight(
                name="rezero_alpha",
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )

    def call(self, inputs, mask=None, training=False):
        """Extracts a single embedding vector for a set of embeddings

        Args:
            inputs: A tensor with shape (batch_size, input_length, hidden_size)
            mask: A boolean tensor with shape (batch_size, input_length, 1).
              All positions with False will be ignored during self attention
            training: Defaults to False.

        Returns:
            Tensor with shape (batch_size, hidden_size)
        """
        attention_mask = mask
        if mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=self.compute_dtype)
            attention_mask = tf.transpose(attention_mask, perm=[0, 2, 1])

        shape = tf.shape(inputs)
        batch_dim = shape[0]
        query = tf.broadcast_to(self.query, shape=[batch_dim, 1, self.hidden_dim])
        output = self.attention(
            query, inputs, attention_mask=attention_mask, training=False
        )

        if self.use_residual_connections:
            print("Pooler residual connection...")
            output = self.query + self._rezero * output

        if self.normalize_output:
            print("Pooler Normalizing outputs...")
            output = self.norm(output)

            if self.compute_dtype == "float16":
                # output_tensor will always be float32
                # so we need to cast it back to float16
                output = tf.cast(output, dtype=tf.float16)
        print("Pooler exit...", self.trainable)
        return tf.squeeze(output, axis=1)

    def get_config(self):
        config = super(MultiHeadAttentionPoolingV2, self).get_config()
        config.update(
            {
                "normalize_output": self.normalize_output,
                "num_heads": self.num_heads,
                "use_residual_connections": self.use_residual_connections,
                "use_linear_bias": self.use_linear_bias,
            }
        )
        return config
