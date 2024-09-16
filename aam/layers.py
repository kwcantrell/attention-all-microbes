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
        self.asv_token = self.num_tokens - 1

        self.nucleotide_position = tf.range(0, 4 * 150, 4, dtype=tf.int32)

    def call(self, inputs, nuc_mask=None, training=False):
        seq = inputs

        if nuc_mask is not None:
            seq = seq * nuc_mask

        # add <ASV> token
        seq = tf.pad(seq, [[0, 0], [0, 0], [0, 1]], constant_values=self.asv_token)

        output = self.emb_layer(seq)
        output = self.avs_attention(output, training=training)

        # set all embeddings that represent pads to 0
        padded_inputs = tf.pad(
            inputs, [[0, 0], [0, 0], [0, 1]], constant_values=self.asv_token
        )
        asv_mask = float_mask(padded_inputs, dtype=self.compute_dtype)
        asv_mask = tf.expand_dims(asv_mask, axis=-1)
        output = output * asv_mask

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
            activation="relu",
            dropout_rate=0.1,
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
            asv_embeddings, attention_mask=attention_mask, training=training
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
    def __init__(self, hidden_dim, num_heads, num_layers, dropout, **kwargs):
        super(NucleotideAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.epsilon = 0.000001

        self.compress_df = tf.keras.layers.Dense(64)
        self.decompress_df = tf.keras.layers.Dense(128)
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
        attention_input = self.compress_df(attention_input)
        attention_input = attention_input + self.pos_emb(attention_input)
        for layer_idx in range(self.num_layers):
            attention_input = self.attention_layers[layer_idx](
                attention_input, training=training
            )

        attention_input = self.decompress_df(attention_input)
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
