import tensorflow as tf
import tenserflow_models as tfm

@tf.keras.saving.register_keras_serializable(
    package="NucleotideEmbedding"
)
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
            dff,
            dropout_rate,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.token_dim = token_dim
        self.max_bp = max_bp
        self.pca_hidden_dim = pca_hidden_dim
        self.pca_heads = pca_heads
        self.pca_layers = pca_layers
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.random_generator = (
            tf.random.Generator.from_non_deterministic_state()
        )

        self.emb_layer = tf.keras.layers.Embedding(
            8,
            512,
            input_length=max_bp,
            embeddings_initializer=tf.keras.initializers.GlorotNormal()
        )
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(
                max_length=max_bp,
                seq_axis=2
        )
        self.pca_layer = PCAProjector(
            hidden_dim=128,
            num_heads=8,
            num_layers=pca_layers,
            dropout=dropout_rate
        )
        self.attention_layer = tfm.nlp.models.TransformerEncoder(
            num_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
            dropout_rate=dropout_rate,
            norm_first=True,
            activation='relu',
        )

    def build(self, input_shape):
        self.nucleotides = tf.Variable(
            tf.zeros(
                shape=(self.max_bp, 6),
                dtype=tf.float32
            ),
            trainable=False,
            dtype=tf.float32
        )

        def _add_random_sequences_and_mask(
            seq,
            seeds,
            rclr,
            unormalized_log_prob,
            range
        ):
            seq_mask = tf.cast(
                tf.math.equal(
                    seq,
                    0
                ),
                dtype=tf.int32
            )

            uniform_mask = tf.random.stateless_uniform(
                tf.shape(seq),
                tf.cast(seeds[:, 0], dtype=tf.int32),
                minval=0.,
                maxval=1.,
                dtype=tf.float32
            )
            nucleotides = tf.reduce_sum(
                tf.multiply(
                    tf.one_hot(
                        seq,
                        depth=6,
                        axis=-1,
                        dtype=tf.float32
                    ),
                    tf.expand_dims(uniform_mask, axis=-1)
                ),
                axis=[0, 1]
            )
            unormalized_log_prob = tf.add(
                tf.nn.softmax(unormalized_log_prob, axis=0),
                tf.nn.softmax(nucleotides, axis=0)
            )
            # unormalized_log_prob += nucleotides

            nuc_strings = tf.multiply(
                tf.reshape(
                    tf.transpose(
                        tf.random.stateless_categorical(
                            unormalized_log_prob,
                            tf.cast(range, dtype=tf.int32),
                            tf.cast(seeds[:, 1], dtype=tf.int32),
                            dtype=tf.int32
                        )
                    ),
                    tf.shape(seq)
                ),
                seq_mask
            )
            seq = tf.concat([seq, nuc_strings[:, -15:, :]], axis=1)
            seq = tf.pad(
                seq,
                [
                    [0, 0],
                    [0, 1],
                    [0, 0]
                ],
                constant_values=6
            )

            rclr_noise = tf.divide(
                tf.random.stateless_normal(
                    tf.shape(nuc_strings[:, -15:, 0]),
                    tf.cast(seeds[:, 2], dtype=tf.int32),
                ),
                tf.cast(
                    tf.reduce_prod(tf.shape(rclr)[1]),
                    dtype=tf.float32
                )
            )
            rclr = tf.concat([rclr, rclr_noise], axis=1)
            rclr = tf.pad(
                rclr,
                paddings=[
                    [0, 0],
                    [0, 1]
                ],
                constant_values=0
            )
            random_mask = tf.random.stateless_binomial(
                tf.shape(seq),
                seeds[:, 3],
                tf.ones_like(
                    seq,
                    dtype=tf.float32
                ),
                [1 - .1],
                output_dtype=tf.int32
            )
            unmasked_sequence = seq
            seq = tf.multiply(
                seq,
                random_mask
            )
            return (seq, unmasked_sequence, rclr, unormalized_log_prob)
        self._add_random_sequences_and_mask = tf.function(
            _add_random_sequences_and_mask,
        )

    def call(self, inputs, training=None):
        seq, rclr = tf.nest.flatten(inputs, expand_composites=True)
        if training:
            range = tf.cast(
                tf.reduce_prod(
                    tf.shape(seq)[:-1],
                ),
                dtype=tf.float32
            )
            seeds = self.random_generator.make_seeds(4)
            unormalized_log_prob = self.nucleotides.read_value()
            seq = tf.cast(seq, dtype=tf.int32)
            seq, unmasked_sequences, rclr, unormalized_log_prob = self._add_random_sequences_and_mask(
                seq,
                seeds,
                rclr,
                unormalized_log_prob,
                range
            )
            seq = tf.cast(seq, dtype=tf.int64)
            self.nucleotides.assign(
                unormalized_log_prob,
                read_value=False
            )
        else:
            seq = tf.pad(
                seq,
                [
                    [0, 0],
                    [0, 1],
                    [0, 0]
                ],
                constant_values=6
            )
            unmasked_sequences = seq

            rclr = tf.pad(
                rclr,
                paddings=[
                    [0, 0],
                    [0, 1]
                ],
                constant_values=0
            )

        mask = tf.cast(
            tf.not_equal(
                seq,
                0
            ),
            dtype=tf.float32
        )
        attention_mask = tf.cast(
            tf.matmul(
                mask,
                mask,
                transpose_b=True
            ),
            dtype=tf.bool
        )
        output = self.emb_layer(seq)
        output = output + self.pos_emb(output)

        output = self.pca_layer(output, training=training, mask=mask)
        output = tf.add(
            output,
            tf.expand_dims(rclr, axis=-1),
        )

        output = self.attention_layer(
            output,
            attention_mask=attention_mask,
            training=training
        )
        return (output, unmasked_sequences[:, :-1, :])

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
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(
    package="PCAProjector"
)
class PCAProjector(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            num_layers,
            dropout,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.pca_layer = MultiHeadPCAProjection(
            hidden_dim,
            num_heads,
            dropout
        )
        self.ff = tf.keras.layers.Dense(
            hidden_dim,
            activation='relu',
            use_bias=True
        )
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.point = tf.keras.layers.Dense(1)
        self.point2 = tf.keras.layers.Dense(
            1,
            activation='relu',
            use_bias=True
        )
        self.ff2 = tf.keras.layers.Dense(
            hidden_dim,
            activation='relu',
            use_bias=True
        )

    def call(self, inputs, mask=None):
        inputs = self.ff(inputs)
        pca_output = self.pca_layer(inputs)
        pca_output = tf.squeeze(self.point(pca_output), axis=-1)
        output = self.dropout_layer(pca_output)
        return output

    def get_config(self):
        base_config = super().get_config()

        config = {
            "hidden_dim":  self.hidden_dim,
            "num_heads":  self.num_heads,
            "num_layers":  self.num_layers,
            "dropout":  self.dropout,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(
    package="MultiHeadPCAProjection"
)
class MultiHeadPCAProjection(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            dropout,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm = tf.keras.layers.LayerNormalization(axis=-2)
        self.norm2 = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        shape = [x if x is not None else -1 for x in input_shape]
        # occurs after up scaling
        head_size = self.hidden_dim // self.num_heads

        self.linear_up_scale = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.hidden_dim,
                activation='relu',
                use_bias=True
            ),
            tf.keras.layers.Dense(
                head_size,
                activation='relu',
                use_bias=True
            )
        ])

        reshape = shape[:-1] + [self.num_heads, head_size]
        first_transp = [i for i in range(len(reshape))]
        first_transp = first_transp[:-3] + [first_transp[-2],
                                            first_transp[-3],
                                            first_transp[-1]]
        second_transp = [i for i in range(len(reshape))]
        second_transp = second_transp[:-3] + [second_transp[-2],
                                              second_transp[-3],
                                              second_transp[-1]]
        second_reshape = shape[:-2] + [self.hidden_dim, head_size]
        init_tup = (
            reshape,
            first_transp,
            second_transp,
            second_reshape,
        )
        self.second = second_transp
        self.compute_proj = MultiHeadPCAProjection.init_proj(*init_tup)

    def init_proj(
        reshape,
        first_transp,
        second_transp,
        second_reshape
    ):
        @tf.function(jit_compile=True)
        def compute_proj(X):
            X = tf.reshape(X, shape=reshape)
            X = tf.transpose(X, perm=first_transp)
            cov = tf.linalg.matmul(X, X, transpose_a=True)
            eig_values, eig_vec = tf.linalg.eigh(cov)
            pca = tf.transpose(
                tf.matmul(
                    tf.linalg.diag(
                        eig_values
                    ),
                    eig_vec,
                ),
                perm=second_transp
            )
            proj = tf.math.softmax(
                tf.divide(
                    pca,
                    tf.math.sqrt(
                        tf.cast(
                            tf.shape(X)[-2],
                            dtype=tf.float32
                        )
                    )
                ),
                axis=-1
            )

            proj = tf.reshape(proj, shape=second_reshape)
            pca = tf.reshape(pca, shape=second_reshape)
            return (proj, pca)
        return compute_proj

    def call(self, inputs):
        output = self.norm(inputs)
        output, pca = self.compute_proj(output)
        output = self.linear_up_scale(output)
        output = self.norm2(output + pca)
        return output

    def get_config(self):
        base_config = super().get_config()
        config = {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(
    package="ReadHead"
)
class ReadHead(tf.keras.layers.Layer):
    def __init__(
            self,
            max_bp,
            hidden_dim,
            num_heads,
            num_layers,
            output_dim,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.max_bp = max_bp
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.read_head = tf.keras.Sequential([
            tf.keras.layers.Dense(
                128,
                activation='relu',
                use_bias=True
            ),
            tf.keras.layers.Dense(
                32,
                activation='relu',
                use_bias=True
            ),
            tf.keras.layers.LayerNormalization(),
        ])
        self.reg_out = tf.keras.layers.Dense(
            self.output_dim,
            use_bias=True
        )
        self.attention_out = tf.keras.layers.Dense(
            max_bp,
            use_bias=True
        )
        self.nuc_out = tf.keras.layers.Dense(
            6,
            use_bias=True
        )

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

    def call(self, inputs, training=None):
        output = self.read_head(inputs)
        reg_out = self.reg_out(output[:, -1, :])
        attention_out = self.nuc_out(
            tf.expand_dims(
                self.attention_out(output[:, :-1, :]),
                axis=-1
            )
        )

        return (reg_out, attention_out)