import tensorflow as tf


class MultiHeadAttentionPooling(tf.keras.layers.Layer):
    def __init__(self, normalize_outputs):
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = 4
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=tf.float32)
        self.normalize_outputs = normalize_outputs

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        key_dim = int(hidden_dim // self.num_heads)
        self.attention = tf.keras.layers.MultiHeadAttention(
            self.num_heads,
            key_dim=key_dim,
            dropout=0.1,
        )

    def get_config(self):
        config = super(MultiHeadAttentionPooling, self).get_config()
        config.update({"normalize_outputs": self.normalize_outputs})
        return config

    def call(self, inputs, mask=None, training=False):
        if mask is not None:
            mask = tf.matmul(mask, mask, transpose_b=True)
        attention = self.attention(
            inputs, inputs, attention_mask=mask, training=training
        )
        if self.normalize_outputs:
            attention = self.norm(attention)

            if self.compute_dtype == "float16":
                # output_tensor will always be float32
                # so we need to cast it back to float16
                attention = tf.cast(attention, dtype=tf.float16)
        return tf.reduce_mean(attention, axis=1)
