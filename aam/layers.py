import tensorflow as tf
import tensorflow_models as tfm


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

        self.emb_layer = tf.keras.layers.Embedding(
            8,
            512,
            input_length=max_bp,
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
            mask_zero=True
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

    def call(self, inputs, training=None):
        seq, rclr = tf.nest.flatten(inputs, expand_composites=True)
        seq = tf.pad(
            seq,
            [
                [0, 0],
                [0, 1],
                [0, 0]
            ],
            constant_values=6
        )
        rclr = tf.pad(
            rclr,
            paddings=[
                [0, 0],
                [0, 1]
            ],
            constant_values=1
        )
        mask = tf.cast(
            tf.expand_dims(
                tf.not_equal(rclr, 0),
                axis=-1
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
        output = tf.multiply(
            output,
            tf.expand_dims(rclr, axis=-1)
        )

        output = self.attention_layer(
            output,
            attention_mask=attention_mask,
            training=training
        )
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

        residual_output = tf.squeeze(self.point(inputs), axis=-1)
        residual_output = self.ff2(residual_output)
        output = self.norm(residual_output + pca_output)
        if mask is not None:
            output = tf.multiply(
                output,
                mask
            )
        output = self.dropout_layer(output)
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

    def build(self, input_shape):
        shape = [x if x is not None else -1 for x in input_shape]
        self.linear_up_scale = tf.keras.layers.Dense(
            self.hidden_dim,
            activation='relu',
            use_bias=True
        )
        # occurs after up scaling
        head_size = self.hidden_dim // self.num_heads
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
            proj = tf.math.softmax(
                tf.divide(
                    tf.transpose(
                        tf.matmul(
                            tf.linalg.diag(
                                eig_values
                            ),
                            eig_vec,
                        ),
                        perm=second_transp
                    ),
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
            return proj
        return compute_proj

    def call(self, inputs):
        output = self.norm(inputs)
        output = self.compute_proj(output)
        output = self.linear_up_scale(output)
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
            hidden_dim,
            num_heads,
            num_layers,
            output_dim,
            **kwargs
    ):
        super().__init__(**kwargs)
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
            150,
            use_bias=True
        )
        self.nuc_out = tf.keras.layers.Dense(
            6,
            use_bias=True
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
        }
        return {**base_config, **config}

    def call(self, inputs, training=None):
        output = self.read_head(inputs, training=training)
        reg_out = self.reg_out(output[:, -1, :])
        attention_out = self.nuc_out(
            tf.expand_dims(
                self.attention_out(output[:, :-1, :]),
                axis=-1
            )
        )

        return (reg_out, attention_out)
