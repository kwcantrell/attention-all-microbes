import tensorflow as tf


class MultiHeadAttentionPooling(tf.keras.layers.Layer):
    def __init__(self):
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = 4

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        key_dim = int(hidden_dim // self.num_heads)
        self.attention = tf.keras.layers.MultiHeadAttention(
            self.num_heads,
            key_dim=key_dim,
            dropout=0.1,
        )

    def call(self, inputs, mask=None, training=False):
        if mask is not None:
            mask = tf.matmul(mask, mask, transpose_b=True)
        attention = self.attention(
            inputs, inputs, attention_mask=mask, training=training
        )
        return tf.reduce_mean(attention, axis=1)
