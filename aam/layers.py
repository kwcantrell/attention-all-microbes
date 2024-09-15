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


@tf.keras.saving.register_keras_serializable(package="NucleotideEmbedding")
class NucleotideEmbedding(tf.keras.layers.Layer):
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
        super(NucleotideEmbedding, self).__init__(**kwargs)
        self.token_dim = token_dim
        self.max_bp = max_bp
        self.pca_hidden_dim = pca_hidden_dim
        self.pca_heads = pca_heads
        self.pca_layers = pca_layers
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.attention_ff = attention_ff
        self.dropout_rate = dropout_rate

        self.base_tokens = 6
        self.num_tokens = self.base_tokens * 150 + 2
        self.emb_layer = tf.keras.layers.Embedding(
            self.num_tokens,
            self.token_dim,
            input_length=self.max_bp,
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.avs_attention = NucleotideAttention(
            128, num_heads=2, num_layers=3, dropout=0.0
        )

        self.asv_pos = tfm.nlp.layers.PositionEmbedding(500, name="asv_pos")
        self.sample_attention = tfm.nlp.models.TransformerEncoder(
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=self.attention_ff,
            norm_first=True,
            activation="relu",
            dropout_rate=0.0,
            intermediate_dropout=0.0,
        )
        self.sample_embedding = self.add_weight(
            "sample_embedding",
            [1, 1, self.token_dim],
            dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )

    def call(self, inputs, return_nuc_attention=True, training=False):
        seq, counts = inputs

        ### Nucleotide level mask ###
        asv_mask = tf.reduce_min(seq, axis=-1, keepdims=True)
        asv_mask = float_mask(asv_mask, dtype=self.compute_dtype)

        # randomly mask 2% in each ASV
        if self.trainable and training:
            seq_mask = tf.random.uniform(
                tf.shape(seq), minval=0, maxval=1, dtype=tf.float32
            )
            seq_mask = tf.less_equal(seq_mask, 0.98)
            seq_mask = tf.cast(seq_mask, dtype=tf.int32)
            seq = tf.multiply(seq, seq_mask)

        seq = tf.pad(seq, [[0, 0], [0, 0], [0, 1]], constant_values=self.num_tokens - 1)
        output = self.emb_layer(seq)
        nuc_attention = self.avs_attention(output, training=training)
        output = nuc_attention[:, :, -1, :]
        output = output * asv_mask

        # sample level embedding vector
        output_shape = tf.shape(output)
        batch_len = output_shape[0]
        emb_len = output_shape[-1]
        sample_emb_shape = [1 for _ in output.get_shape().as_list()]
        sample_emb_shape[0] = batch_len
        sample_emb_shape[-1] = emb_len
        sample_embedding = tf.broadcast_to(self.sample_embedding, sample_emb_shape)
        output = tf.concat([output, sample_embedding], axis=1)

        ### Sample level mask ###
        attention_mask = tf.pad(asv_mask, [[0, 0], [0, 1], [0, 0]], constant_values=1)
        attention_mask = tf.matmul(attention_mask, attention_mask, transpose_b=True)
        attention_mask = tf.cast(attention_mask, dtype=tf.bool)

        asv_pos = self.asv_pos(output)
        output = output + asv_pos
        attention_output = self.sample_attention(
            output, attention_mask=attention_mask, training=training
        )

        if return_nuc_attention:
            return attention_output[:, -1, :], nuc_attention[:, :, :-1, :]
        else:
            return attention_output, counts

    def sequence_embedding(self, seq):
        seq = tf.pad(seq, [[0, 0], [0, 0], [0, 1]], constant_values=self.num_tokens - 1)
        output = self.emb_layer(seq)
        output = self.avs_attention(output)
        return output[:, :, -1, :]

    def get_config(self):
        config = super(NucleotideEmbedding, self).get_config()
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
    def __init__(self, hidden_dim, num_heads, num_layers, dropout, **kwargs):
        super(NucleotideAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.epsilon = 0.000001

        self.pos_emb = tfm.nlp.layers.PositionEmbedding(
            max_length=151, seq_axis=2, name="nuc_pos"
        )
        self.attention_layers = []
        for i in range(self.num_layers):
            self.attention_layers.append(
                NucleotideAttentionBlock(
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    epsilon=self.epsilon,
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
        return tf.cast(output, dtype=self.compute_dtype)

    def get_config(self):
        config = super(NucleotideAttention, self).get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            }
        )
        return config


@tf.keras.saving.register_keras_serializable(package="NucleotideAttentionBlock")
class NucleotideAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, dropout, epsilon=0.000001, **kwargs):
        super(NucleotideAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = dropout
        self.epsilon = epsilon

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        self.head_size = int(input_shape[-1] / self.num_heads)

        # scaled dot product sublayer
        def add_transformation_weights(name):
            return (
                self.add_weight(
                    f"{name}_dense",
                    [1, 1, 1, self.hidden_dim, self.head_size],
                    dtype=tf.float32,
                ),
                self.add_weight(
                    f"w_{name}i",
                    [1, 1, self.num_heads, self.head_size, self.head_size],
                    dtype=tf.float32,
                ),
            )

        self.q_dense, self.w_qi = add_transformation_weights("q")
        self.k_dense, self.w_ki = add_transformation_weights("k")
        self.v_dense, self.w_vi = add_transformation_weights("v")

        self.scale_dot_factor = tf.cast(
            tf.math.sqrt(float(self.head_size)), dtype=self.compute_dtype
        )
        self.attention_dropout = tf.keras.layers.Dropout(self.dropout)
        self.attention_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype=tf.float32
        )

        self.multihead_value_einsum = "sahnj,sahjk->sanhk"
        self.o_dense = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)

        def scaled_dot_attention(attention_input):
            # linear projections for query, key, and value
            def compute_wi(attention_input, dense, w):
                transformed_input = tf.expand_dims(attention_input, axis=2)
                dense_output = tf.matmul(transformed_input, dense)
                wi_output = tf.matmul(dense_output, w)
                return wi_output

            wq_tensor = compute_wi(attention_input, self.q_dense, self.w_qi)
            wk_tensor = compute_wi(attention_input, self.k_dense, self.w_ki)
            wv_tensor = compute_wi(attention_input, self.v_dense, self.w_vi)

            # (multihead) scaled dot product attention sublayer
            dot_tensor = tf.linalg.matmul(wq_tensor, wk_tensor, transpose_b=True)
            scaled_dot_tensor = tf.divide(dot_tensor, self.scale_dot_factor)
            softmax_tensor = tf.keras.activations.softmax(scaled_dot_tensor, axis=-2)
            attention_output = tf.einsum(
                "sahij,sahjk->saihk", softmax_tensor, wv_tensor
            )

            # reshape
            batch_size = tf.shape(attention_input)[0]
            attention_output = tf.reshape(
                attention_output,
                shape=[batch_size, -1, 151, self.num_heads * self.head_size],
            )
            attention_output = self.o_dense(attention_output)
            return attention_output

        attention_input_shape = list(input_shape)
        attention_input_shape[0] = None
        attention_input_shape[-2] = 151
        self.scaled_dot_attention = tf.function(
            scaled_dot_attention,
            input_signature=[
                tf.TensorSpec(shape=attention_input_shape, dtype=self.compute_dtype)
            ],
            reduce_retracing=True,
        )

        self.ff_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype=tf.float32
        )
        self.inter_ff = tf.keras.layers.Dense(128, activation="relu")
        self.outer_ff = tf.keras.layers.Dense(self.num_heads * self.head_size)
        self.ff_dropout = tf.keras.layers.Dropout(self.dropout)

        super(NucleotideAttentionBlock, self).build(attention_input_shape)

    def call(self, attention_input, batch_size=8, training=False):
        # scaled dot product attention sublayer
        attention_input = self.attention_norm(attention_input)
        attention_input = tf.cast(attention_input, dtype=self.compute_dtype)

        attention_output = self.scaled_dot_attention(attention_input)
        attention_output = self.attention_dropout(attention_output, training=training)

        ff_input = attention_input + attention_output
        ff_output = self.ff_norm(ff_input)
        ff_output = self.inter_ff(ff_output)
        ff_output = self.outer_ff(ff_output)
        return self.ff_dropout(ff_output + ff_input)

    def get_config(self):
        config = super(NucleotideAttentionBlock, self).get_config()

        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "epsilon": self.epsilon,
            }
        )

        return config


@tf.keras.saving.register_keras_serializable(package="TransferAttention")
class TransferAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dropout_rate=0.0,
        hidden_dim=128,
        **kwargs,
    ):
        super(TransferAttention, self).__init__(**kwargs)
        self.epsilon = 0.000001
        self.num_heads = 4
        self.hidden_dim = hidden_dim
        self.dropout = dropout_rate
        self.head_size = int(self.hidden_dim / self.num_heads)

        self.feature_multihead_attention = tf.keras.layers.MultiHeadAttention(
            self.num_heads, hidden_dim, use_bias=False, dropout=self.dropout
        )
        self.feature_rank = tfm.nlp.layers.PositionEmbedding(500)

        self.count_multihead_attention = tf.keras.layers.MultiHeadAttention(
            self.num_heads, hidden_dim, use_bias=True, dropout=self.dropout
        )
        self.count_inter_ff = tf.keras.layers.Dense(1024, use_bias=True)
        self.count_outer_ff = tf.keras.layers.Dense(128, activation="relu")
        self.count_rank = tfm.nlp.layers.PositionEmbedding(500)
        self.count_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype=tf.float32
        )

        self.ff_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype=tf.float32
        )
        self.inter_ff = tf.keras.layers.Dense(128, activation="relu")
        self.outer_ff = tf.keras.layers.Dense(self.num_heads * self.head_size)

        self.encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            dropout_rate=self.dropout,
        )
        self.reg_in = tf.keras.layers.Dense(128, activation="relu")
        self.reg_out = tf.keras.layers.Dense(1, use_bias=True)
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training=False):
        features, counts = inputs

        # # convert to rel_abundance
        # counts = counts / tf.reduce_sum(counts, axis=-1, keepdims=True)

        # get attention mask
        attention_mask = float_mask(counts)
        attention_mask = tf.pad(attention_mask, [[0, 0], [0, 1]], constant_values=1)
        attention_mask = tf.expand_dims(attention_mask, axis=-1)
        attention_mask = tf.matmul(attention_mask, attention_mask, transpose_b=True)

        # add pad to account for sample token
        counts = tf.pad(counts, [[0, 0], [0, 1]], constant_values=0)
        counts = tf.expand_dims(counts, axis=-1)

        # shift tensors such that sample token is at the front of tensor
        counts = tf.roll(counts, shift=1, axis=1)
        features = tf.roll(features, shift=1, axis=1)
        attention_mask = tf.roll(attention_mask, shift=1, axis=1)

        # project counts
        counts = self.count_inter_ff(counts)
        counts = self.count_outer_ff(counts)
        # counts = tf.cast(self.count_norm(counts), dtype=self.compute_dtype)

        # add rank
        counts = counts + self.count_rank(counts)
        features = features + self.feature_rank(features)

        # compute attention
        count_value_attention = self.count_multihead_attention(
            counts,
            counts,
            return_attention_scores=False,
            attention_mask=attention_mask,
            training=training,
        )
        feature_value_attention = self.feature_multihead_attention(
            features,
            features,
            return_attention_scores=False,
            attention_mask=attention_mask,
            training=training,
        )
        attention_output = count_value_attention + feature_value_attention

        # send attention_output through ff network
        normalize_attention = tf.cast(
            self.ff_norm(attention_output), dtype=self.compute_dtype
        )
        normalize_attention = self.inter_ff(normalize_attention)
        normalize_attention = self.outer_ff(normalize_attention)

        final_attention = attention_output + normalize_attention
        final_attention = self.dropout_layer(final_attention, training=training)

        # regression on sample embedding
        reg_out = self.encoder(
            # final_attention, attention_mask=attention_mask, training=training
            count_value_attention + features,
            attention_mask=attention_mask,
            training=training,
        )
        reg_out = self.reg_in(reg_out[:, 0, :])
        reg_out = self.reg_out(reg_out)
        return reg_out

    def get_config(self):
        config = super(TransferAttention, self).get_config()
        config.update({"dropout_rate": self.dropout_rate, "d_model": self.d_model})
        return config


@tf.keras.saving.register_keras_serializable(package="ReadHead")
class ReadHead(tf.keras.layers.Layer):
    def __init__(
        self,
        max_bp,
        hidden_dim,
        num_heads,
        num_layers,
        output_dim,
        output_nuc=True,
        reg_ff=1024,
        **kwargs,
    ):
        super(ReadHead, self).__init__(**kwargs)
        self.max_bp = max_bp
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.output_nuc = output_nuc

        if output_nuc:
            self.att_ff = tf.keras.layers.Dense(6)
            self.softmax = tf.keras.layers.Activation("softmax", dtype=tf.float32)

    def call(self, inputs):
        reg_out, seq_out = inputs

        if self.output_nuc:
            seq_out = self.att_ff(seq_out)
            seq_out = self.softmax(seq_out)

        return reg_out, seq_out

    #     self.reg_inter_ff = tf.keras.layers.Dense(128, activation="relu")
    #     self.reg_out_ff = tf.keras.layers.Dense(128)
    #
    #     if output_nuc:
    #         self.att_inter_ff = tf.keras.layers.Dense(128, activation="relu")
    #         self.att_out_ff = tf.keras.layers.Dense(128, activation="relu")
    #         self.seq_logits = tf.keras.layers.Dense(6)
    #         self.softmax = tf.keras.layers.Activation("softmax", dtype=tf.float32)
    #
    # def call(self, inputs):
    #     reg_out, seq_out = inputs
    #
    #     reg_out = self.reg_inter_ff(reg_out)
    #     reg_out = self.reg_out_ff(reg_out)
    #
    #     if self.output_nuc:
    #         seq_out = self.att_inter_ff(seq_out)
    #         seq_out = self.att_out_ff(seq_out)
    #         seq_out = self.seq_logits(seq_out)
    #         seq_out = self.softmax(seq_out)
    #
    #     return reg_out, seq_out

    def sequence_logits(self, sequence_embeddings):
        seq_out = tf.reshape(
            self.attention_out(sequence_embeddings),
            shape=tf.concat(
                [tf.shape(sequence_embeddings)[:-1], [self.max_bp, 1]], axis=0
            ),
        )
        seq_out = seq_out + self.pos_emb(seq_out)
        seq_out = self.pos_norm(seq_out)
        seq_out = self.nuc_out(seq_out)
        return seq_out

    def get_config(self):
        config = super(ReadHead, self).get_config()
        config.update(
            {
                "max_bp": self.max_bp,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "output_dim": self.output_dim,
                "output_nuc": self.output_nuc,
                "reg_ff": self.reg_ff,
            }
        )
        return config
