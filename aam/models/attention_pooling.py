import tensorflow as tf


class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionPooling, self).__init__()
        self.query = tf.keras.layers.Dense(1, use_bias=False)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None, training=False):
        # Compute attention scores
        attention_scores = self.query(inputs)  # [B, T, 1]
        attention_scores = tf.squeeze(attention_scores, axis=-1)  # [B, T]

        # Scale the scores for numerical stability
        attention_scores = attention_scores / tf.sqrt(tf.cast(tf.shape(inputs)[-1], tf.float32))

        # Apply mask (if provided)
        if mask is not None:
            mask = tf.squeeze(mask, axis=-1)
            mask = tf.cast(mask, dtype=tf.float32)  # [B, T]
            attention_scores += (1.0 - mask) * -1e9  # Mask padding tokens

        # Compute attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # [B, T]

        # Weighted sum (pooling)
        pooled_output = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, axis=-1), axis=1)  # [B, D]

        # Apply normalization
        return self.norm(pooled_output)
