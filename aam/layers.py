import tensorflow as tf
import tensorflow_models as tfm

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
        normalize_outputs=True,
        use_residual_connections=False,
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
        self.base_tokens = 6
        self.num_tokens = self.base_tokens * self.max_bp + 2
        self.normalize_outputs = normalize_outputs
        self.use_residual_connections = use_residual_connections

        self.asv_token = self.num_tokens - 1
        self.nucleotide_position = tf.range(0, self.base_tokens * self.max_bp, self.base_tokens, dtype=tf.int32)
        self.nuc_loss = tf.keras.losses.CategoricalCrossentropy(reduction="none")

        # nuc postions start at 1 as 0 is used for mask token
        # self.nuc_pred = tf.keras.layers.Dense(5, activation="softmax")
        self.nuc_pred = tf.keras.layers.Dense(self.base_tokens * self.max_bp, use_bias=True, dtype=tf.float32)
        self._softmax = tf.keras.layers.Activation("softmax", dtype=tf.float32)

    def build(self, input_shape):
        self.emb_layer = tf.keras.layers.Embedding(
            self.num_tokens,
            self.embedding_dim,
            input_length=self.max_bp,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
        )

        self._rezero = self.add_weight(name="rezero_alpha", initializer="zeros", trainable=True, dtype=tf.float32)
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(
            self.max_bp + 1, seq_axis=1, initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
        )

        self.asv_attention = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            dropout_rate=self.dropout_rate,
            intermediate_size=self.intermediate_ff,
            activation=self.intermediate_activation,
            normalize_outputs=False,
            use_residual_connections=self.use_residual_connections,
        )
        self.norm_out = tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-6, dtype=tf.float32)
        super(ASVEncoder, self).build(input_shape)

    def call(self, inputs, include_bert_random_mask=True, training=False):
        inputs = tf.cast(inputs, dtype=tf.int32)
        inputs_shape = tf.shape(inputs)

        # boolean mask used to select non-pad tokens
        mask = tf.reduce_sum(inputs, axis=-1) > 0  # shape [B, A]
        mask = tf.reshape(mask, shape=[-1])  # shape [B * A]

        # mask for non-pad tokens (used during the creation of random_mask)
        valid_mask = tf.cast(inputs > 0, dtype=tf.int32)

        # select 15% of tokens to "mask" i.e. tokens to use to compute nuc_loss
        random_mask = create_random_mask(inputs_shape, percent=0.02, dtype=tf.int32) * valid_mask
        masked_inputs = inputs
        if include_bert_random_mask and training:
            # of the masked tokens, select 20% to either keep or change to
            # random token
            random_non_mask = create_random_mask(inputs_shape, percent=0.2, dtype=tf.int32) * random_mask

            # of the 20% of masked tokens to either keep or change, select 50%  to keep
            # and 50% to change
            random_change = create_random_mask(inputs_shape, percent=0.5, dtype=tf.int32)

            # tokens to keep the same
            random_keep = random_non_mask * random_change

            # tokens to randomly change
            random_change = (1 - random_keep) * valid_mask * random_non_mask

            # step 1: change all random_mask positions to <MASK> token
            masked_input = masked_inputs * (1 - random_mask)

            # step 2: change 10% of <MASK> tokens back to original token
            masked_input = masked_input + masked_inputs * random_keep * random_mask * valid_mask

            # step 3: change 10% of <MASK> tokens to random token
            random_tokens = tf.random.uniform(tf.shape(masked_inputs), minval=1, maxval=4, dtype=tf.int32)

            # step 4: create masked input
            masked_input = masked_input + random_tokens * random_change * random_mask * valid_mask
            masked_inputs = masked_input

        # convert random_mask to boolean mask
        random_mask = random_mask > 0

        # get nucleotides embeddigns
        asv_tokens = masked_inputs + self.nucleotide_position
        asv_input = self.emb_layer(asv_tokens)
        asv_input = asv_input + tf.cast(self._rezero, dtype=self.compute_dtype) * self.pos_emb(asv_input)

        output = self.asv_attention(asv_input, training=training)
        output = self.norm_out(output, training=training)

        # extract the masked nucleotides
        unmasked_tokens = inputs + self.nucleotide_position
        masked_nuc = tf.reshape(random_mask, shape=[-1])
        nuc_embeddings = tf.reshape(output, shape=[-1, self.embedding_dim])
        unmasked_tokens = tf.reshape(unmasked_tokens, shape=[-1])[masked_nuc]
        masked_nuc = nuc_embeddings[masked_nuc]
        nuc_pred = self._softmax(self.nuc_pred(masked_nuc))
        self._compute_nuc_loss(unmasked_tokens, nuc_pred)
        print("ASVEncoder exit...")
        return output

    def _compute_nuc_loss(self, tokens, pred):
        tokens = tf.one_hot(tokens, tf.shape(pred)[-1])
        nuc_loss = self.nuc_loss(tokens, pred)
        nuc_loss = tf.reduce_mean(nuc_loss)
        self.add_loss(tf.reduce_mean(nuc_loss))

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
                "normalize_outputs": self.normalize_outputs,
                "use_residual_connections": self.use_residual_connections,
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
        attention_mask = tf.pad(attention_mask, [[0, 0], [0, 1], [0, 0]], constant_values=1)
        attention_mask = tf.matmul(attention_mask, attention_mask, transpose_b=True)

        sample_embeddings = self.sample_attention(asv_embeddings, attention_mask=attention_mask > 0, training=training)
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
        self.output_normalization = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, dtype=tf.float32)
        super(NucleotideAttention, self).build(input_shape)

    def call(self, attention_input, attention_mask=None, training=False):
        attention_input = attention_input + self.pos_emb(attention_input)
        attention_input = attention_input  # * (9 * 3) ** (-0.25)
        for layer_idx in range(self.num_layers):
            attention_input = self.attention_layers[layer_idx](attention_input, training=training)
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

        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, dtype=tf.float32)
        self.attention_dropout = tf.keras.layers.Dropout(self.dropout)
        self.ff_dropout = tf.keras.layers.Dropout(self.dropout)
        self.ff_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, dtype=tf.float32)
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
        self.o_dense = self.add_weight("w_o", wo_shape, trainable=True, dtype=tf.float32)
        # self.o_dense = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)

        self.scale_dot_factor = tf.math.sqrt(tf.cast(self.head_size, dtype=self.compute_dtype))

        self.inter_ff = tf.keras.layers.Dense(self.intermediate_ff, activation=self.intermediate_activation, use_bias=True)
        self.outer_ff = tf.keras.layers.Dense(self.hidden_dim, use_bias=True)

        super(NucleotideAttentionBlock, self).build(input_shape)

    # linear projections for query, key, and value
    def compute_wi(self, attention_input, w):
        # [B, A, N, E] => [B, A, 1, N, E] * [1,1,H,E,S]
        transformed_input = tf.expand_dims(attention_input, axis=2)

        # [B, A, 1, N, E] => [B, A, H, N, S]
        wi_output = tf.matmul(transformed_input, tf.cast(w, dtype=self.compute_dtype))
        transformed_input = tf.ensure_shape(wi_output, [None, None, self.num_heads, self.nucleotides, self.head_size])
        return wi_output

    def scaled_dot_attention(self, attention_input):
        wq_tensor = self.compute_wi(attention_input, self.w_qi)
        wk_tensor = self.compute_wi(attention_input, self.w_ki)
        wv_tensor = self.compute_wi(attention_input, self.w_vi)  # * (0.67 * 3) ** -0.25)

        # (multihead) scaled dot product attention sublayer
        # [B, A, H, N, S] => [B, A, H, N, N]
        dot_tensor = tf.linalg.matmul(wq_tensor, wk_tensor, transpose_b=True)
        dot_tensor = tf.ensure_shape(dot_tensor, [None, None, self.num_heads, self.nucleotides, self.nucleotides])

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
