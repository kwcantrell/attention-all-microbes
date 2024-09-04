import tensorflow as tf
import tensorflow_models as tfm

from aam.sequence_utils import (
    add_seq_and_count_pad,
    sequence_attention_mask,
)
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
        super().__init__(**kwargs)
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
        polyval=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_dim = token_dim
        self.max_bp = max_bp
        self.pca_hidden_dim = pca_hidden_dim
        self.pca_heads = pca_heads
        self.pca_layers = pca_layers
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.attention_ff = attention_ff
        self.dropout_rate = dropout_rate
        self.random_generator = tf.random.Generator.from_non_deterministic_state()

        self.base_tokens = 6
        self.num_tokens = self.base_tokens * 150 + 2
        self.emb_layer = tf.keras.layers.Embedding(
            self.num_tokens,
            self.token_dim,
            input_length=self.max_bp,
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.avs_attention = NucleotideAttention(128, num_heads=2, num_layers=3, dropout=0.0)

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
        self.add_seq_and_count_pad = tf.function(add_seq_and_count_pad)
        self.sequence_attention_mask = tf.function(sequence_attention_mask)

    def call(self, inputs, return_nuc_attention=True, training=False):
        seq, _ = inputs

        token_adjustment = tf.reshape(tf.range(self.max_bp), shape=[1, 1, self.max_bp]) * self.base_tokens
        seq = seq + token_adjustment
        # if self.trainable and training:
        #     seq_mask = tf.random.uniform(
        #         tf.shape(seq), minval=0, maxval=1, dtype=tf.float32
        #     )
        #     seq_mask = tf.greater(seq_mask, 0.9)
        #     seq_mask = tf.cast(seq_mask, dtype=tf.int32)
        #     seq = tf.multiply(seq, 1 - seq_mask)
        seq = tf.pad(seq, [[0, 0], [0, 0], [0, 1]], constant_values=self.num_tokens - 1)

        output = self.emb_layer(seq)
        nuc_attention = self.avs_attention(output, training=training)

        output = nuc_attention[:, :, -1, :]

        ### Masking ASVs ###
        # if the first "nucleotide" is zero then the entire ASV was a pad
        asv_mask = seq[:, :, :1]
        asv_mask = tf.reduce_max(asv_mask, axis=-1, keepdims=True)
        asv_mask = tf.cast(asv_mask, dtype=self.compute_dtype)
        output = output * asv_mask
        ### Nucleotide Level ###

        batch_dim = tf.shape(output)[0]
        output = tf.concat(
            [output, tf.tile(self.sample_embedding, (batch_dim, 1, 1))],
            axis=1,
        )
        ### Mask ASV ###
        attention_mask = tf.pad(asv_mask, [[0, 0], [0, 1], [0, 0]], constant_values=1)
        attention_mask = tf.matmul(attention_mask, attention_mask, transpose_b=True)
        attention_mask = tf.cast(attention_mask, dtype=tf.bool)
        ### Sample Level ###

        attention_output = self.sample_attention(output, attention_mask=attention_mask, training=training)

        if return_nuc_attention:
            return (attention_output[:, -1, :], nuc_attention[:, :, :-1, :])
        else:
            return attention_output

    def sequence_embedding(self, seq):
        seq = tf.pad(seq, [[0, 0], [0, 0], [0, 1]], constant_values=self.num_tokens - 1)
        output = self.emb_layer(seq)
        output = self.avs_attention(output)
        return output[:, :, -1, :]

    def get_config(self):
        base_config = super().get_config()
        config = {
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
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(package="NucleotideAttention")
class NucleotideAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.attention_layers = []
        for i in range(self.num_layers):
            self.attention_layers.append(
                NucleotideAttentionBlock(
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    name=("layer_%d" % i),
                )
            )

    def build(self, input_shape):
        # token for asv
        self.emd_dim = 64
        self.ff = tf.keras.layers.Dense(self.emd_dim, use_bias=False)

        # position encoding sublayer
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(max_length=151, seq_axis=2)

        # output sublayer
        self.ff_output = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)

    def call(self, attention_input, attention_mask=None, training=False):
        # add asv token
        attention_input = self.ff(attention_input)
        attention_input = attention_input + self.pos_emb(attention_input)

        for layer_idx in range(self.num_layers):
            attention_input = self.attention_layers[layer_idx](attention_input, training=training)
        # extract asv token
        nucleotide_output = self.ff_output(attention_input)
        return nucleotide_output


@tf.keras.saving.register_keras_serializable(package="NucleotideAttentionBlock")
class NucleotideAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = dropout
        self.singular_values = 8

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

        self.scale_dot_factor = tf.cast(tf.math.sqrt(float(self.head_size)), dtype=self.compute_dtype)
        self.attention_dropout = tf.keras.layers.Dropout(self.dropout)
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=0.0001)

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
            attention_output = tf.einsum("sahij,sahjk->saihk", softmax_tensor, wv_tensor)

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
            input_signature=[tf.TensorSpec(shape=attention_input_shape, dtype=self.compute_dtype)],
            reduce_retracing=True,
        )
        print(attention_input_shape)

        self.ff = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(epsilon=0.0001),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(self.num_heads * self.head_size),
            ]
        )
        self.ff_dropout = tf.keras.layers.Dropout(self.dropout)

        super().build(attention_input_shape)

    def call(self, attention_input, batch_size=8, training=False):
        # scaled dot product attention sublayer
        attention_input = self.attention_norm(attention_input)
        attention_output = self.scaled_dot_attention(attention_input)
        attention_output = self.attention_dropout(attention_output, training=training)

        ff_input = attention_input + attention_output
        ff_output = self.ff(ff_input) + ff_input
        return self.ff_dropout(ff_output)

    def get_config(self):
        base_config = super().get_config()

        config = {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(package="TransferAttention")
class TransferAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        dropout_rate=0,
        d_model=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_attention = tfm.nlp.models.TransformerEncoder(
            num_layers=4,
            num_attention_heads=4,
            dropout_rate=0.0,
            intermediate_size=1024,
            norm_first=True,
            activation="relu",
            name="transfer_attention",
        )
        self.reg_out = tf.keras.layers.Dense(1, use_bias=True)

    def build(self, input_shape):
        seq_shape, _ = input_shape
        # hidden_dim = seq_shape[-1]
        # self.count_ff = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Dense(1024, use_bias=True, activation="relu"),
        #         tf.keras.layers.Dense(hidden_dim, use_bias=True),
        #         tf.keras.layers.LayerNormalization(),
        #         tf.keras.layers.Dropout(0.0),
        #     ]
        # )

    def call(self, inputs, training=False):
        attention_input, rel_count = inputs

        attention_mask = tf.pad(rel_count, [[0, 0], [0, 1]], constant_values=1)
        attention_mask = tf.expand_dims(attention_mask, axis=-1)
        attention_mask = float_mask(
            tf.matmul(attention_mask, attention_mask, transpose_b=True),
            dtype=self.compute_dtype,
        )

        rel_count = tf.expand_dims(rel_count, axis=-1)
        # rel_count = self.count_ff(rel_count)
        rel_count = tf.pad(rel_count, [[0, 0], [0, 1], [0, 0]], constant_values=0)

        attention_input = attention_input + rel_count
        attention_input = self.sample_attention(attention_input, attention_mask=attention_mask, training=training)

        reg_out = self.reg_out(attention_input[:, -1, :])
        return reg_out


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
        super().__init__(**kwargs)
        self.max_bp = max_bp
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.output_nuc = output_nuc

        if output_nuc:
            self.attention_out = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(1024, use_bias=True, activation="relu"),
                    tf.keras.layers.Dense(128, use_bias=True),
                    tf.keras.layers.LayerNormalization(epsilon=0.0001),
                    tf.keras.layers.Dense(6),
                ]
            )

    def call(self, inputs):
        reg_out, seq_out = inputs

        if self.output_nuc:
            seq_out = self.attention_out(seq_out)

        return (reg_out, seq_out)

    def sequence_logits(self, sequence_embeddings):
        seq_out = tf.reshape(
            self.attention_out(sequence_embeddings),
            shape=tf.concat([tf.shape(sequence_embeddings)[:-1], [self.max_bp, 1]], axis=0),
        )
        seq_out = seq_out + self.pos_emb(seq_out)
        seq_out = self.pos_norm(seq_out)
        seq_out = self.nuc_out(seq_out)
        return seq_out

    def get_config(self):
        base_config = super().get_config()
        config = {
            "max_bp": self.max_bp,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
        }
        return {**base_config, **config}
