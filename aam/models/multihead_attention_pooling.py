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
        # # Compute attention scores
        # attention_scores = self.query(inputs)  # [B, T, H]

        # # Scale the scores for numerical stability
        # attention_scores = attention_scores / tf.sqrt(
        #     tf.cast(tf.shape(inputs)[-1], self.compute_dtype)
        # )

        # # Apply mask (if provided)
        # if mask is not None:
        #     # negative number (dtypes.float16.min) is divided by 2, in order to
        #     # avoid overflows when summing negative inputs.
        #     if self.compute_dtype == tf.float16:
        #         large_neg_num = tf.float16.min / 2.0
        #     else:
        #         large_neg_num = -1e9

        #     mask = tf.cast(mask, dtype=self.compute_dtype)  # [B, T, 1]
        #     attention_scores += (1.0 - mask) * large_neg_num  # Mask padding tokens

        # # Compute attention weights
        # attention_scores = tf.transpose(attention_scores, perm=[0, 2, 1])  # [B, H, T]
        # attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        # attention_weights = self.dropout(attention_weights, training=training)
        # attention_weights = tf.expand_dims(attention_weights, axis=-1)  # [B, H, T, 1]

        # inputs = tf.expand_dims(inputs, axis=1)  # [B, 1, T, D]
        # pooled_output = tf.reduce_sum(inputs * attention_weights, axis=2)  # [B, H, D]

        # output_tensor = tf.reduce_mean(pooled_output, axis=1)
        # if self.output_norm:
        #     # Apply normalization
        #     output_tensor = self.norm(output_tensor)

        #     if self.compute_dtype == "float16":
        #         # output_tensor will always be float32
        #         # so we need to cast it back to float16
        #         output_tensor = tf.cast(output_tensor, dtype=tf.float16)
        # return output_tensor
