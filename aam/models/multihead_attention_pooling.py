import tensorflow as tf

from aam.models.attention_pooling import AttentionPooling


class MultiHeadAttentionPooling(tf.keras.layers.Layer):
    def __init__(self):
        super(MultiHeadAttentionPooling, self).__init__()
        self.query = tf.keras.layers.Dense(8, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pool = AttentionPooling()

    def call(self, inputs, mask=None, training=False):
        # Compute attention scores
        attention_scores = self.query(inputs)  # [B, T, H]

        # Scale the scores for numerical stability
        attention_scores = attention_scores / tf.sqrt(
            tf.cast(tf.shape(inputs)[-1], tf.float32)
        )

        # Apply mask (if provided)
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.float32)  # [B, T, 1]
            attention_scores += (1.0 - mask) * -1e9  # Mask padding tokens

        # Compute attention weights
        attention_scores = tf.transpose(attention_scores, perm=[0, 2, 1])  # [B, H, T]
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)  # [B, H, T, 1]

        inputs = tf.expand_dims(inputs, axis=1)  # [B, 1, T, D]
        pooled_output = self.norm(
            tf.reduce_sum(inputs * attention_weights, axis=2)
        )  # [B, H, D]

        # Apply normalization
        return self.pool(pooled_output)
