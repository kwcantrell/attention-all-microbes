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

        self.emb_layer = tf.keras.layers.Embedding(
            8,
            self.token_dim,
            input_length=self.max_bp,
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
        )

        self.pca_layer = NucleotideAttention(
            128, num_heads=1, num_layers=3, dropout=0.02
        )

        self.attention_layer = tfm.nlp.models.TransformerEncoder(
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=self.attention_ff,
            norm_first=True,
            activation="relu",
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

    def call(self, inputs, return_attention=False):
        seq, _ = inputs
        seq = tf.pad(seq, [[0, 0], [0, 0], [0, 1]], constant_values=6)
        output = self.emb_layer(seq)
        output = self.pca_layer(output)

        batch_dim = tf.shape(output)[0]
        output = tf.concat(
            [output, tf.tile(self.sample_embedding, (batch_dim, 1, 1))],
            axis=1,
        )

        seq, rclr = tf.nest.flatten(self.add_seq_and_count_pad(inputs))
        attention_mask = self.sequence_attention_mask(seq[:, :, :150])
        attention_output = self.attention_layer(output, attention_mask=attention_mask)
        print(self.variable_dtype, self.compute_dtype)
        if self.variable_dtype != self.compute_dtype:
            attention_output = tf.cast(attention_output, self.compute_dtype)
        if return_attention:
            return attention_output
        else:
            return tf.concat([output[:, :-1, :], attention_output[:, -1:, :]], axis=1)

    def sequence_embedding(self, seq):
        seq = tf.pad(seq, [[0, 0], [0, 0], [0, 1]], constant_values=6)
        output = self.emb_layer(seq)
        output = self.pca_layer(output)
        return output

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
                    dff_dim=128,
                    dropout=self.dropout,
                    name=("layer_%d" % i),
                )
            )

    def build(self, input_shape):
        # token for asv
        self.emd_dim = 64
        self.ff = tf.keras.layers.Dense(self.emd_dim, use_bias=False)
        asv_token_shape = [1 for _ in input_shape]
        asv_token_shape[-1] = self.emd_dim
        self.extend_sequence = [[0, 0] for _ in input_shape]
        self.extend_sequence[-2] = [0, 1]

        # position encoding sublayer
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(max_length=151, seq_axis=2)
        self.pos_norm = tf.keras.layers.LayerNormalization()

        # output sublayer
        self.ff_output = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)
        self.output_norm = tf.keras.layers.LayerNormalization()

    def call(self, attention_input, training=False):
        # add asv token
        attention_input = attention_input + self.pos_emb(attention_input)
        attention_input = self.ff(attention_input)

        # position encoding sublayer
        # attention_input = self.pos_norm(pos_embedded_tensor)

        for layer_idx in range(self.num_layers):
            attention_input = self.attention_layers[layer_idx](
                attention_input, training=training
            )
        # extract asv token
        nucleotide_output = attention_input[:, :, -1, :]
        nucleotide_output = self.ff_output(nucleotide_output)
        # nucleotide_output = self.output_norm(nucleotide_output)
        return nucleotide_output


@tf.keras.saving.register_keras_serializable(package="NucleotideAttentionBlock")
class NucleotideAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, dff_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dff_dim = dff_dim
        self.dropout = dropout
        self.singular_values = 8

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        self.head_size = int(input_shape[-1] / self.num_heads)

        # scaled dot product sublayer
        def add_transformation_weights(name):
            return (
                self.add_weight(f"{name}_dense", [self.hidden_dim, self.head_size]),
                self.add_weight(
                    f"w_{name}i", [self.num_heads, self.head_size, self.head_size]
                ),
            )

        self.q_dense, self.w_qi = add_transformation_weights("q")
        self.k_dense, self.w_ki = add_transformation_weights("k")
        self.v_dense, self.w_vi = add_transformation_weights("v")

        self.scale_dot_factor = tf.math.sqrt(float(self.head_size))
        self.attention_dropout = tf.keras.layers.Dropout(self.dropout)
        self.attention_norm = tf.keras.layers.LayerNormalization()

        self.multihead_value_einsum = "sahnj,sahjk->sanhk"
        self.o_dense = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)

        def scaled_dot_attention(attention_input):
            # linear projections for query, key, and value
            def compute_wi(attention_input, dense, w):
                wi = tf.matmul(dense, w)
                wi = tf.reshape(
                    wi, shape=[1, 1, self.num_heads, self.hidden_dim, self.head_size]
                )
                transformed_input = tf.expand_dims(attention_input, axis=2)
                return tf.matmul(transformed_input, wi)

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
                tf.TensorSpec(shape=attention_input_shape, dtype=tf.float32)
            ],
            reduce_retracing=True,
        )
        print(attention_input_shape)

        super().build(attention_input_shape)

    def call(self, attention_input, batch_size=8, training=False):
        # scaled dot product attention sublayer
        attention_input = self.attention_norm(attention_input)
        attention_output = self.scaled_dot_attention(attention_input)
        attention_output = self.attention_dropout(attention_output)
        return attention_input + attention_output

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
        # self.attention_layer = tfm.nlp.models.TransformerEncoder(
        #     num_layers=4,
        #     num_attention_heads=4,
        #     dropout_rate=0.0,
        #     attention_dropout_rate=0.0,
        #     intermediate_dropout=0.0,
        #     intermediate_size=256,
        #     norm_first=True,
        #     activation="relu",
        #     name="transfer_attention",
        # )
        self.attention_layer = tf.keras.layers.MultiHeadAttention(
            4,
            key_dim=32,
            dropout=0.,
            use_bias=False,

        )
        self.attention_reg = ActivityRegularizationLayer(
            tf.keras.regularizers.L2(0.001)
        )
        self.add_seq_and_count_pad = tf.function(add_seq_and_count_pad)
        self.sequence_attention_mask = tf.function(sequence_attention_mask)

    def call(self, inputs, training=False):
        attention_input, rel_count = inputs
        rel_count = tf.expand_dims(rel_count, axis=-1)
        attention_mask = tf.cast(tf.matmul(rel_count, rel_count, transpose_b=True), dtype=tf.bool)
        embeddings = tf.multiply(attention_input, rel_count)
        output = self.attention_layer(
            query=embeddings, value=embeddings, key=embeddings, attention_mask=attention_mask, training=training
        )
        return output


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
                    tf.keras.layers.Dense(max_bp * 64, use_bias=True, activation="relu")
                ]
            )
            self.pos_emb = tfm.nlp.layers.PositionEmbedding(
                max_length=max_bp,
                seq_axis=2,
            )
            self.pos_norm = tf.keras.layers.LayerNormalization()
            self.nuc_out = tf.keras.layers.Dense(6, use_bias=True)

    def call(self, inputs):
        reg_out = inputs[:, -1, :]
        seq_out = inputs[:, :-1, :]
        
        if self.output_nuc:
            batch_dim = tf.shape(inputs)[0]
            num_seq = tf.shape(inputs)[1] - 1

            seq_out = self.attention_out(seq_out)
            seq_out = tf.reshape(
                seq_out,
                shape=[batch_dim, num_seq, self.max_bp, 64],
            )
            seq_out = seq_out + self.pos_emb(seq_out)

            seq_out = self.nuc_out(seq_out)

        return (reg_out, seq_out)

    def sequence_logits(self, sequence_embeddings):
        seq_out = tf.reshape(
            self.attention_out(sequence_embeddings),
            shape=tf.concat(
                [tf.shape(sequence_embeddings)[:-1], [self.max_bp, 1]], axis=0
            ),
        )
        seq_out = seq_out + self.pos_emb(seq_out)
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


@tf.keras.saving.register_keras_serializable(package="ReadHead2")
class ReadHead2(tf.keras.layers.Layer):
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
        
        if output_dim==1:
            print('???')
            self.reg_out = tf.keras.layers.Dense(1)

        if output_nuc:
            self.attention_out = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(max_bp * 64, use_bias=True, activation="relu")
                ]
            )
            self.pos_emb = tfm.nlp.layers.PositionEmbedding(
                max_length=max_bp,
                seq_axis=2,
            )
            self.pos_norm = tf.keras.layers.LayerNormalization()
            self.nuc_out = tf.keras.layers.Dense(6, use_bias=True)

    def call(self, inputs):
        reg_out = inputs[:, -1, :]
        seq_out = inputs[:, :-1, :]
        
        if self.output_dim:
            reg_out = self.reg_out(reg_out)
            
        if self.output_nuc:
            batch_dim = tf.shape(inputs)[0]
            num_seq = tf.shape(inputs)[1] - 1

            seq_out = self.attention_out(seq_out)
            seq_out = tf.reshape(
                seq_out,
                shape=[batch_dim, num_seq, self.max_bp, 64],
            )
            seq_out = seq_out + self.pos_emb(seq_out)

            seq_out = self.nuc_out(seq_out)

        return (reg_out, seq_out)

    def sequence_logits(self, sequence_embeddings):
        seq_out = tf.reshape(
            self.attention_out(sequence_embeddings),
            shape=tf.concat(
                [tf.shape(sequence_embeddings)[:-1], [self.max_bp, 1]], axis=0
            ),
        )
        seq_out = seq_out + self.pos_emb(seq_out)
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
