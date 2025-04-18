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
        include_pos_emb=False,
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
        self.num_tokens = 5

        print(f"create asv layer with {self.attention_heads} heads")
        self.asv_token = self.num_tokens - 1
        self.nuc_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            ignore_class=0, reduction=tf.keras.losses.Reduction.NONE
        )

        self.randomize = 0.97
        self.rand_nucs = 0.03
        self.nucs_to_obs = 0.15
        self.emb_layer = tf.keras.layers.Embedding(
            self.num_tokens, self.embedding_dim, input_length=self.max_bp
        )
        self.include_pos_emb = include_pos_emb
        if self.include_pos_emb:
            print("including pos embeddings")
            self.pos_emb = tfm.nlp.layers.PositionEmbedding(
                self.max_bp, initializer="uniform"
            )

        self.asv_attention = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=self.intermediate_ff,
            activation=self.intermediate_activation,
        )
        self.nuc_pred = tf.keras.Sequential(
            [
                FeedForward(),
                tf.keras.layers.Dense(self.num_tokens),
                tf.keras.layers.Activation("softmax", dtype=tf.float32),
            ]
        )

    def build(self, input_shape):
        print("Building ASVEncoder...")
        self._build_input_shape = input_shape

        self.emb_layer.build(input_shape)

        input_shape = self.emb_layer.compute_output_shape(input_shape)
        self.asv_attention.build(input_shape)

        input_shape = self.asv_attention.compute_output_shape(input_shape)
        self.nuc_pred.build(input_shape)
        self.built = True
        print("ASVEncoder built!")

    def compute_output_shape(self, input_shape):
        return input_shape + (self.embedding_dim,)

    def _mask_nucs(self, inputs, random_mask):
        return inputs * (1 - random_mask)

    def _random_nucs(self, inputs, random_mask):
        shape = tf.shape(inputs)
        random_nucs = tf.random.uniform(shape, minval=0, maxval=5, dtype=tf.int32)
        masked_inputs = self._mask_nucs(inputs, random_mask)
        return masked_inputs + random_nucs * random_mask

    def _observe_first_and_last_positions(self, obs_mask):
        """Sets the first and last position of each sequence in obs_mask to 1."""
        shape = tf.shape(obs_mask)
        batch_dim = shape[0]
        last_i = shape[-1] - 1
        batch = tf.expand_dims(tf.range(batch_dim, dtype=tf.int32), axis=-1)
        first_pos = tf.expand_dims(tf.zeros(batch_dim, dtype=tf.int32), axis=-1)
        last_pos = tf.expand_dims(tf.ones(batch_dim, dtype=tf.int32) * last_i, axis=-1)
        first_indices = tf.concat([batch, first_pos], axis=-1)
        last_indices = tf.concat([batch, last_pos], axis=-1)
        indices = tf.concat([first_indices, last_indices], axis=0)
        mask = tf.scatter_nd(indices, tf.ones(2 * batch_dim, dtype=tf.int32), shape)
        obs_mask = tf.cast(obs_mask, dtype=tf.int32)
        return (obs_mask + mask) > 0

    def call(self, inputs, return_randomize=False, training=False):
        training = training and self.trainable
        inputs = tf.cast(inputs, dtype=tf.int32)

        input_shape = tf.shape(inputs)

        # randomize tokens
        randomize = tf.cast(
            tf.random.uniform([input_shape[0], 1]) < self.randomize, dtype=tf.int32
        )
        random_mask = create_random_mask(input_shape, self.rand_nucs, dtype=tf.int32)
        random_tokens = self._random_nucs(inputs, random_mask * randomize)
        if training:
            emb_inputs = random_tokens
        else:
            emb_inputs = inputs

        # compute embeddings
        asv_input = self.emb_layer(emb_inputs)
        if self.include_pos_emb:
            asv_input = asv_input + self.pos_emb(asv_input)
            asv_input = asv_input * tf.cast(0.5, dtype=self.compute_dtype)

        output = self.asv_attention(asv_input, training=training)

        if self.trainable:
            # compute cross entropy on 10 percent of nucleotides
            obs_mask = create_random_mask(input_shape, self.nucs_to_obs, dtype=tf.int32)
            obs_mask = obs_mask + random_mask
            obs_mask = self._observe_first_and_last_positions(obs_mask)
            loss = self._compute_nuc_loss(inputs, output, obs_mask)
            self.add_loss(loss)

        print("ASVEncoder exit...", self.trainable)
        if not return_randomize:
            return output
        else:
            randomize = tf.where(tf.squeeze(randomize, axis=-1) < 1)
            return output, randomize

    def _compute_nuc_loss(self, tokens, embeddings, mask):
        shape = tf.shape(mask)
        batch_dim = shape[0]
        seq_dim = shape[-1]

        counts = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1, keepdims=True)
        counts = tf.repeat(counts, repeats=seq_dim, axis=-1)
        counts = counts * tf.cast(batch_dim, dtype=tf.float32)
        counts = tf.cast(1.0, dtype=tf.float32) / counts

        tokens = tokens[mask]
        counts = counts[mask]
        embeddings = embeddings[mask]

        nuc_preds = self.nuc_pred(embeddings)
        loss = self.nuc_loss(tokens, nuc_preds) * counts
        return tf.reduce_sum(loss)

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
                "include_pos_emb": self.include_pos_emb,
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
