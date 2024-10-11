import tensorflow as tf
import tensorflow_models as tfm

from aam.utils import float_mask


@tf.keras.saving.register_keras_serializable(package="activity_regularization")
class ActivityRegularizationLayer(tf.keras.layers.Layer):
    def __init__(self, reg):
        super().__init__()
        self.reg = reg

    def call(self, inputs, reg_mask=None):
        reg = inputs
        if reg_mask is not None:
            reg = tf.multiply(reg, tf.expand_dims(float_mask(reg_mask), axis=-1))
        self.add_loss(self.reg(reg))
        return inputs


@tf.keras.saving.register_keras_serializable(package="InputLayer")
class InputLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InputLayer, self).__init__(**kwargs)
        self.trainable = False

    def call(self, inputs):
        return inputs


@tf.keras.saving.register_keras_serializable(package="ASVEncoder")
class ASVEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        token_dim,
        max_bp,
        pca_hidden_dim,
        pca_heads,
        pca_layers,
        attention_heads,
        attention_layers,
        attention_ff,
        dropout_rate,
        intermediate_ff,
        **kwargs,
    ):
        super(ASVEncoder, self).__init__(**kwargs)
        self.token_dim = token_dim
        self.max_bp = max_bp
        self.pca_hidden_dim = pca_hidden_dim
        self.pca_heads = pca_heads
        self.pca_layers = pca_layers
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.attention_ff = attention_ff
        self.dropout_rate = dropout_rate
        self.intermediate_ff = intermediate_ff

        self.base_tokens = 6
        self.num_tokens = self.base_tokens * self.max_bp + 2
        self.emb_layer = tf.keras.layers.Embedding(
            self.num_tokens,
            self.token_dim,
            input_length=self.max_bp,
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.avs_attention = NucleotideAttention(
            128,
            max_bp=self.max_bp,
            num_heads=self.attention_heads,
            num_layers=self.attention_layers,
            dropout=self.dropout_rate,
            intermediate_ff=intermediate_ff,
        )
        self.asv_token = self.num_tokens - 1

        self.nucleotide_position = tf.range(
            0, self.base_tokens * self.max_bp, self.base_tokens, dtype=tf.int32
        )

    def call(self, inputs, seq_mask=None, training=False):
        seq = inputs
        seq = seq + self.nucleotide_position

        # add <ASV> token
        seq = tf.pad(seq, [[0, 0], [0, 0], [0, 1]], constant_values=self.asv_token)

        output = self.emb_layer(seq)
        output = self.avs_attention(output, training=training)

        return output

    def sequence_embedding(self, seq):
        seq = tf.pad(seq, [[0, 0], [0, 0], [0, 1]], constant_values=self.asv_token)
        output = self.emb_layer(seq)
        output = self.avs_attention(output)
        return output[:, :, -1, :]

    def get_config(self):
        config = super(ASVEncoder, self).get_config()
        config.update(
            {
                "token_dim": self.token_dim,
                "max_bp": self.max_bp,
                "pca_hidden_dim": self.pca_hidden_dim,
                "pca_heads": self.pca_heads,
                "pca_layers": self.pca_layers,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "attention_ff": self.attention_ff,
                "dropout_rate": self.dropout_rate,
                "intermediate_ff": self.intermediate_ff,
            }
        )
        return config


@tf.keras.saving.register_keras_serializable(package="SampleEncoder")
class SampleEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        token_dim,
        max_bp,
        pca_hidden_dim,
        pca_heads,
        pca_layers,
        attention_heads,
        attention_layers,
        attention_ff,
        dropout_rate,
        **kwargs,
    ):
        super(SampleEncoder, self).__init__(**kwargs)
        dropout_rate = dropout_rate
        self.token_dim = token_dim
        self.max_bp = max_bp
        self.pca_hidden_dim = pca_hidden_dim
        self.pca_heads = pca_heads
        self.pca_layers = pca_layers
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.attention_ff = attention_ff
        self.dropout_rate = dropout_rate

        self.sample_attention = tfm.nlp.models.TransformerEncoder(
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=self.attention_ff,
            norm_first=True,
            activation="silu",
            intermediate_dropout=self.dropout_rate,
        )
        self.sample_token = self.add_weight(
            "sample_token",
            [1, 1, self.token_dim],
            dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )

    def call(self, inputs, attention_mask=None, training=False):
        # add <SAMPLE> token empbedding
        asv_shape = tf.shape(inputs)
        batch_len = asv_shape[0]
        emb_len = asv_shape[-1]
        sample_emb_shape = [1 for _ in inputs.get_shape().as_list()]
        sample_emb_shape[0] = batch_len
        sample_emb_shape[-1] = emb_len
        sample_token = tf.broadcast_to(self.sample_token, sample_emb_shape)
        asv_embeddings = tf.concat([inputs, sample_token], axis=1)

        # extend mask to account for <SAMPLE> token
        attention_mask = tf.pad(
            attention_mask, [[0, 0], [0, 1], [0, 0]], constant_values=1
        )
        attention_mask = tf.matmul(attention_mask, attention_mask, transpose_b=True)

        sample_embeddings = self.sample_attention(
            asv_embeddings, attention_mask=attention_mask > 0, training=training
        )
        return sample_embeddings

    def get_config(self):
        config = super(SampleEncoder, self).get_config()
        config.update(
            {
                "token_dim": self.token_dim,
                "max_bp": self.max_bp,
                "pca_hidden_dim": self.pca_hidden_dim,
                "pca_heads": self.pca_heads,
                "pca_layers": self.pca_layers,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "attention_ff": self.attention_ff,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@tf.keras.saving.register_keras_serializable(package="NucleotideAttention")
class NucleotideAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        max_bp,
        num_heads,
        num_layers,
        dropout,
        intermediate_ff=1024,
        **kwargs,
    ):
        super(NucleotideAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.max_bp = max_bp
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.epsilon = 0.000001
        self.intermediate_ff = intermediate_ff
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(513, name="nuc_pos")
        self.attention_layers = []
        for i in range(self.num_layers):
            self.attention_layers.append(
                NucleotideAttentionBlock(
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    epsilon=self.epsilon,
                    intermediate_ff=intermediate_ff,
                    name=("layer_%d" % i),
                )
            )
        self.output_normalization = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype=tf.float32
        )

    def call(self, attention_input, attention_mask=None, training=False):
        attention_input = attention_input + self.pos_emb(attention_input)
        for layer_idx in range(self.num_layers):
            attention_input = self.attention_layers[layer_idx](
                attention_input, training=training
            )
        output = self.output_normalization(attention_input)
        return output

    def get_config(self):
        config = super(NucleotideAttention, self).get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "max_bp": self.max_bp,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "intermediate_ff": self.intermediate_ff,
            }
        )
        return config


@tf.keras.saving.register_keras_serializable(package="NucleotideAttentionBlock")
class NucleotideAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self, num_heads, dropout, epsilon=0.000001, intermediate_ff=1024, **kwargs
    ):
        super(NucleotideAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = dropout
        self.epsilon = epsilon
        self.intermediate_ff = intermediate_ff
        self.attention_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype=tf.float32
        )
        self.attention_dropout = tf.keras.layers.Dropout(self.dropout)
        self.ff_dropout = tf.keras.layers.Dropout(self.dropout)
        self.ff_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype=tf.float32
        )

    def build(self, input_shape):
        self._shape = input_shape
        self.nucleotides = input_shape[2]
        self.hidden_dim = input_shape[3]
        self.head_size = tf.cast(self.hidden_dim / self.num_heads, dtype=tf.int32)

        wi_shape = [1, 1, self.num_heads, self.hidden_dim, self.head_size]
        self.w_qi = self.add_weight("w_qi", wi_shape, trainable=True, dtype=tf.float32)
        self.w_ki = self.add_weight("w_ki", wi_shape, trainable=True, dtype=tf.float32)
        self.w_vi = self.add_weight("w_kv", wi_shape, trainable=True, dtype=tf.float32)
        self.o_dense = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)

        self.scale_dot_factor = tf.math.sqrt(
            tf.cast(self.head_size, dtype=self.compute_dtype)
        )

        self.inter_ff = tf.keras.layers.Dense(
            self.intermediate_ff, activation="silu", use_bias=True
        )
        self.outer_ff = tf.keras.layers.Dense(self.hidden_dim, use_bias=True)

        super(NucleotideAttentionBlock, self).build(input_shape)

    # linear projections for query, key, and value
    def compute_wi(self, attention_input, w):
        # [B, A, N, E] => [B, A, 1, N, E] * [1,1,H,E,S]
        transformed_input = tf.expand_dims(attention_input, axis=2)

        # [B, A, 1, N, E] => [B, A, H, N, S]
        wi_output = tf.matmul(transformed_input, tf.cast(w, dtype=self.compute_dtype))
        transformed_input = tf.ensure_shape(
            wi_output, [None, None, self.num_heads, self.nucleotides, self.head_size]
        )
        return wi_output

    def scaled_dot_attention(self, attention_input):
        wq_tensor = self.compute_wi(attention_input, self.w_qi)
        wk_tensor = self.compute_wi(attention_input, self.w_ki)
        wv_tensor = self.compute_wi(attention_input, self.w_vi)

        # (multihead) scaled dot product attention sublayer
        # [B, A, H, N, S] => [B, A, H, N, N]
        dot_tensor = tf.linalg.matmul(wq_tensor, wk_tensor, transpose_b=True)
        dot_tensor = tf.ensure_shape(
            dot_tensor, [None, None, self.num_heads, self.nucleotides, self.nucleotides]
        )

        scaled_dot_tensor = tf.multiply(dot_tensor, 1 / self.scale_dot_factor)
        softmax_tensor = tf.keras.activations.softmax(scaled_dot_tensor, axis=-1)

        # [B, A, H, N, N] => [B, A, H, N, S]
        attention_output = tf.matmul(softmax_tensor, wv_tensor)

        # [B, A, H, N, S] => [B, A, N, H, S]
        attention_output = tf.transpose(attention_output, perm=[0, 1, 3, 2, 4])
        attention_output = tf.ensure_shape(
            attention_output,
            [None, None, self.nucleotides, self.num_heads, self.head_size],
        )
        # reshape
        shape = tf.shape(attention_input)
        batch_size = shape[0]
        num_asv = shape[1]
        attention_output = tf.reshape(
            attention_output,
            shape=[batch_size, num_asv, self.nucleotides, self.hidden_dim],
        )
        attention_output = self.o_dense(attention_output)
        attention_output = tf.ensure_shape(attention_output, self._shape)
        return attention_output

    def call(self, attention_input, training=False):
        # scaled dot product attention sublayer
        attention_input = self.attention_norm(attention_input)

        # cast for mixed precision
        _attention_input = tf.cast(attention_input, dtype=self.compute_dtype)

        # cast back to float32
        _attention_output = self.scaled_dot_attention(_attention_input)
        attention_output = tf.cast(_attention_output, dtype=tf.float32)

        # residual connection
        attention_output = tf.add(attention_input, attention_output)
        attention_output = tf.ensure_shape(attention_output, self._shape)
        # attention_output = self.attention_dropout(attention_output, training=training)

        # cast for mixed precision
        ff_input = self.ff_norm(attention_output)
        _ff_input = tf.cast(ff_input, dtype=self.compute_dtype)
        _ff_output = self.inter_ff(_ff_input)
        _ff_output = self.outer_ff(_ff_output)

        # cast back to float32
        ff_output = tf.cast(_ff_output, dtype=tf.float32)
        ff_output = tf.add(ff_input, ff_output)

        ff_output = tf.ensure_shape(ff_output, self._shape)
        ff_output = self.ff_dropout(ff_output, training=training)
        return ff_output

    def get_config(self):
        config = super(NucleotideAttentionBlock, self).get_config()

        config.update(
            {
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "intermediate_ff": self.intermediate_ff,
            }
        )

        return config
