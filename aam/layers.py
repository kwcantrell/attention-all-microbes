import tensorflow as tf
import tensorflow_models as tfm

from aam.losses import global_embedding_l2_regulization
from aam.models.feedforward import FeedForward
from aam.models.transformers import TransformerEncoder
from aam.utils import create_random_mask, float_mask


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
        max_bp,
        attention_heads,
        attention_layers,
        dropout_rate,
        intermediate_ff,
        intermediate_activation="gelu",
        add_token=True,
        embedding_dim=128,
        use_residual_connections=False,
        use_linear_bias=True,
        **kwargs,
    ):
        super(ASVEncoder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.max_bp = max_bp
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.dropout_rate = dropout_rate
        self.intermediate_ff = intermediate_ff
        self.intermediate_activation = intermediate_activation
        self.add_token = add_token
        self.base_tokens = 5
        self.num_tokens = self.base_tokens
        self.use_residual_connections = use_residual_connections
        self.use_linear_bias = use_linear_bias

        print(f"create asv layer with {self.attention_heads} heads")
        self.asv_token = self.num_tokens - 1
        self.nuc_loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE, label_smoothing=0.05
        )

    def build(self, input_shape):
        if self.built:
            print("ASVEncoder is already built")
            return

        self.emb_layer = tf.keras.layers.Embedding(
            self.num_tokens, self.embedding_dim, input_length=self.max_bp
        )

        if not self.use_linear_bias:
            self._rezero = self.add_weight(
                name="rezero_alpha",
                initializer="zeros",
                trainable=True,
                dtype=tf.float32,
            )
            self.pos_emb = tfm.nlp.layers.PositionEmbedding(self.max_bp + 1, seq_axis=1)

        self.asv_attention = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            dropout_rate=0.0,
            intermediate_size=self.intermediate_ff,
            activation=self.intermediate_activation,
            use_residual_connections=self.use_residual_connections,
            use_linear_bias=self.use_linear_bias,
        )

        self.nuc_pred = tf.keras.Sequential([FeedForward(), tf.keras.layers.Dense(2)])
        self.nuc_output_activation = tf.keras.layers.Activation(
            "softmax", dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs, include_bert_random_mask=True, training=False):
        training = training and self.trainable
        inputs = tf.cast(inputs, dtype=tf.int32)

        input_shape = tf.shape(inputs)
        random_indices = tf.random.shuffle(inputs)

        # 10 percent chance not to randomize sequence
        randomize_sequence = tf.cast(tf.random.uniform([1, 1]) > 0.1, dtype=tf.int32)

        # mark upto 10 percent of the nucleotides to be randomized
        random_mask = (
            create_random_mask(
                input_shape,
                percent=tf.random.uniform([], minval=0.0, maxval=0.1),
                dtype=tf.int32,
            )
            * randomize_sequence
        )

        # compute cross entropy on 10-20 percent
        observe_mask = (
            create_random_mask(
                input_shape,
                percent=tf.random.uniform([], minval=0.1, maxval=0.2),
                dtype=tf.int32,
            )
            + random_mask
        )
        observe_mask = observe_mask > 0

        # zero out  the randomized bloxks
        shuffled_input = inputs * (1 - random_mask)

        # add the randomized indices
        shuffled_input = shuffled_input + random_indices * random_mask

        if training:
            emb_inputs = shuffled_input
        else:
            emb_inputs = inputs

        positions = tf.cast(emb_inputs == inputs, dtype=tf.int32)

        # get nucleotides embeddigns
        asv_input = self.emb_layer(emb_inputs)

        # only add positional embeddings if using vanilla Transforer
        if not self.use_linear_bias:
            asv_input = asv_input + tf.cast(
                self._rezero, dtype=self.compute_dtype
            ) * self.pos_emb(asv_input)

        # pass embeddings through Transformer
        output = self.asv_attention(asv_input, training=training)

        # generate training loss
        loss = self._compute_nuc_loss(positions, output, observe_mask)

        if include_bert_random_mask and self.trainable:
            self.add_loss(loss)

        print("ASVEncoder exit...", self.trainable)
        return output

    def _compute_nuc_loss(self, tokens, embeddings, mask):
        emb_shape = tf.shape(embeddings)
        batch_dim = emb_shape[0]
        seq_dim = emb_shape[1]

        tokens = tokens[mask]
        embeddings = embeddings[mask]
        nuc_pred = self.nuc_output_activation(self.nuc_pred(embeddings))
        tokens = tf.one_hot(tokens, depth=tf.shape(nuc_pred)[-1])
        loss = self.nuc_loss(tokens, nuc_pred)

        loss = tf.scatter_nd(tf.where(mask), loss, [batch_dim, seq_dim])
        loss = tf.math.divide_no_nan(
            tf.reduce_sum(loss, axis=-1),
            tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1),
        )
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super(ASVEncoder, self).get_config()
        config.update(
            {
                "max_bp": self.max_bp,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "dropout_rate": self.dropout_rate,
                "intermediate_ff": self.intermediate_ff,
                "intermediate_activation": self.intermediate_activation,
                "add_token": self.add_token,
                "embedding_dim": self.embedding_dim,
                "use_residual_connections": self.use_residual_connections,
                "use_linear_bias": self.use_linear_bias,
            }
        )
        return config


@tf.keras.saving.register_keras_serializable(package="SampleEncoder")
class SampleEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        token_dim,
        max_bp,
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
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.attention_ff = attention_ff
        self.dropout_rate = dropout_rate

        self.sample_attention = tfm.nlp.models.TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=self.attention_ff,
            norm_first=True,
            activation="relu",
            dropout_rate=self.dropout_rate,
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
        max_bp,
        num_heads,
        num_layers,
        dropout,
        intermediate_ff=1024,
        intermediate_activation="gelu",
        embedding_dim=128,
        **kwargs,
    ):
        super(NucleotideAttention, self).__init__(**kwargs)
        self.max_bp = max_bp
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.epsilon = 1e-6
        self.intermediate_ff = intermediate_ff
        self.intermediate_activation = intermediate_activation
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(
            self.max_bp + 1,
            seq_axis=1,
            # initializer=tf.keras.initializers.RandomNormal(
            #     mean=0, stddev=self.embedding_dim**0.5
            # ),
            name="nuc_pos",
        )
        self.attention_layers = []
        for i in range(self.num_layers):
            self.attention_layers.append(
                NucleotideAttentionBlock(
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    epsilon=self.epsilon,
                    intermediate_ff=self.intermediate_ff,
                    intermediate_activation=self.intermediate_activation,
                    name=("layer_%d" % i),
                )
            )
        self.output_normalization = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype=tf.float32
        )
        super(NucleotideAttention, self).build(input_shape)

    def call(self, attention_input, attention_mask=None, training=False):
        attention_input = attention_input + self.pos_emb(attention_input)
        attention_input = attention_input  # * (9 * 3) ** (-0.25)
        for layer_idx in range(self.num_layers):
            attention_input = self.attention_layers[layer_idx](
                attention_input, training=training
            )
        # output = self.output_normalization(attention_input)
        return attention_input

    def get_config(self):
        config = super(NucleotideAttention, self).get_config()
        config.update(
            {
                "max_bp": self.max_bp,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "intermediate_ff": self.intermediate_ff,
                "intermediate_activation": self.intermediate_activation,
                "embedding_dim": self.embedding_dim,
            }
        )
        return config


@tf.keras.saving.register_keras_serializable(package="NucleotideAttentionBlock")
class NucleotideAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        dropout,
        epsilon=1e-6,
        intermediate_ff=1024,
        intermediate_activation="gelu",
        **kwargs,
    ):
        super(NucleotideAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = dropout
        self.epsilon = epsilon
        self.intermediate_ff = intermediate_ff
        self.intermediate_activation = intermediate_activation

    def build(self, input_shape):
        self._shape = input_shape
        self.nucleotides = input_shape[2]
        self.hidden_dim = input_shape[3]
        self.head_size = tf.cast(self.hidden_dim / self.num_heads, dtype=tf.int32)

        self.attention_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype=tf.float32
        )
        self.attention_dropout = tf.keras.layers.Dropout(self.dropout)
        self.ff_dropout = tf.keras.layers.Dropout(self.dropout)
        self.ff_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype=tf.float32
        )
        self.nuc_alpha = self.add_weight(
            name="nuc_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

        wi_shape = [1, 1, self.num_heads, self.hidden_dim, self.head_size]
        self.w_qi = self.add_weight("w_qi", wi_shape, trainable=True, dtype=tf.float32)
        self.w_ki = self.add_weight("w_ki", wi_shape, trainable=True, dtype=tf.float32)
        self.w_vi = self.add_weight("w_kv", wi_shape, trainable=True, dtype=tf.float32)

        wo_shape = [1, 1, self.hidden_dim, self.hidden_dim]
        self.o_dense = self.add_weight(
            "w_o", wo_shape, trainable=True, dtype=tf.float32
        )
        # self.o_dense = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)

        self.scale_dot_factor = tf.math.sqrt(
            tf.cast(self.head_size, dtype=self.compute_dtype)
        )

        self.inter_ff = tf.keras.layers.Dense(
            self.intermediate_ff, activation=self.intermediate_activation, use_bias=True
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
        wv_tensor = self.compute_wi(
            attention_input, self.w_vi
        )  # * (0.67 * 3) ** -0.25)

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
        attention_output = tf.matmul(
            attention_output,
            self.o_dense,  # * (0.67 * 3) ** (-0.25)
        )
        # attention_output = self.o_dense(attention_output)
        attention_output = tf.ensure_shape(attention_output, self._shape)
        return attention_output

    def call(self, attention_input, training=False):
        # # scaled dot product attention sublayer
        # attention_input = self.attention_norm(attention_input)

        # cast for mixed precision
        _attention_input = tf.cast(attention_input, dtype=self.compute_dtype)

        # cast back to float32
        _attention_output = self.scaled_dot_attention(_attention_input)
        attention_output = tf.cast(_attention_output, dtype=tf.float32) * self.nuc_alpha

        # residual connection
        attention_output = tf.add(attention_input, attention_output)
        attention_output = tf.ensure_shape(attention_output, self._shape)
        attention_output = self.attention_dropout(attention_output, training=training)

        # cast for mixed precision
        # ff_input = self.ff_norm(attention_output)
        ff_input = attention_output  # self.ff_norm(attention_output)
        _ff_input = tf.cast(ff_input, dtype=self.compute_dtype)
        _ff_output = self.inter_ff(_ff_input)
        _ff_output = self.outer_ff(_ff_output)

        # cast back to float32, residual connection
        ff_output = tf.cast(_ff_output, dtype=tf.float32) * self.nuc_alpha
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
                "intermediate_activation": self.intermediate_activation,
            }
        )

        return config
