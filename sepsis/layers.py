import tensorflow as tf
import tensorflow_models as tfm


@tf.keras.saving.register_keras_serializable(
    package="FeatureEmbedding"
)
class FeatureEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        token_dim,
        emb_vocab,
        features_add_rate,
        d_model,
        ff_dim,
        dropout_rate,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_dim = token_dim
        self.emb_vocab = emb_vocab
        self.features_add_rate = features_add_rate
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.tokens = tf.range(
            tf.constant(len(emb_vocab) + 1, dtype=tf.int64)
        )
        self.total_tokens = tf.reduce_max(self.tokens) + 1
        # add one for mask
        self.feature_tokens = tf.keras.layers.StringLookup(
            vocabulary=emb_vocab,
            mask_token='<MASK>',
            num_oov_indices=0,
            output_mode='int',
            name="tokens",
            dtype=tf.int64
        )

        self.embedding_layer = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=len(emb_vocab)+1,
                    output_dim=token_dim,
                    embeddings_initializer="ones"
                ),
                tf.keras.layers.Dense(
                    d_model,
                    activation='relu',
                    use_bias=True
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(dropout_rate)
            ]
        )

        def _component_block():
            return (
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(
                            ff_dim,
                            activation='relu',
                            use_bias=True
                        ),
                        tf.keras.layers.LayerNormalization(),
                        tf.keras.layers.Dropout(dropout_rate)
                    ]
                )
            )
        self.ff_pca = _component_block()
        self.ff_loadings = _component_block()

    def call(self, inputs, training=True):
        feature, rclr = inputs
        feature_tokens = self.feature_tokens(feature)

        # extend feature tensor to add random tokens
        feature_count = tf.multiply(
            tf.reduce_sum(
                tf.cast(
                    feature_tokens > 0,
                    dtype=tf.float32
                ),
                axis=-1,
                keepdims=True,
            ),
            self.features_add_rate
        )

        feature_size = tf.cast(tf.shape(feature_tokens)[1], dtype=tf.int64)
        feature_mask = tf.not_equal(feature_tokens, 0)

        def _random_tokens(inputs):
            sample_tokens, features_to_add = inputs
            features_to_add = tf.squeeze(features_to_add)

            in_sample_mask = tf.scatter_nd(
                tf.expand_dims(sample_tokens, axis=-1),
                tf.ones_like(sample_tokens),
                shape=[self.total_tokens]
            )

            # get random tokens
            random_tokens = tf.random.shuffle(
                self.tokens[in_sample_mask < 1]
            )
            random_tokens = random_tokens[
                :tf.minimum(feature_size, features_to_add)
            ]
            reduced_size = tf.cast(
                tf.shape(random_tokens)[0],
                dtype=tf.int64
            )
            random_tokens = tf.pad(
                random_tokens,
                [[feature_size - reduced_size, 0]]
            )
            return random_tokens

        random_tokens = tf.map_fn(
            _random_tokens,
            (
                feature_tokens,
                tf.cast(feature_count, dtype=tf.int64)
            ),
            fn_output_signature=tf.int64
        )

        # get embedding vector
        sample_tokens = feature_tokens + random_tokens
        output = self.embedding_layer(sample_tokens)

        # add a 1 to rclr where random tokens
        sample_rclr = tf.add(
            rclr,
            tf.cast(
                random_tokens > 0,
                dtype=tf.float32
            )
        )

        # scale embeddings by rclr
        output = tf.multiply(
            output,
            tf.expand_dims(
                sample_rclr,
                axis=-1
            )
        )

        # prep embeddings for objective functions
        output_embeddings = self.ff_loadings(output)
        output_regression = self.ff_pca(output)

        return [
            feature_mask,
            sample_tokens,
            output_embeddings,
            output_regression
        ]

    def get_config(self):
        base_config = super().get_config()
        config = {
            "token_dim": self.token_dim,
            "emb_vocab": self.emb_vocab,
            "features_add_rate": self.features_add_rate,
            "d_model": self.d_model,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(
    package="PCA"
)
class PCA(tf.keras.layers.Layer):
    def __init__(
        self,
        emb_dim,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.flag = self.add_weight(trainable=False)

    def call(self, inputs):
        if self.flag == 0:
            self.flag.assign_add(1)
            output = inputs
        else:
            output = self.pca(inputs)
        return output

    def pca(self, input):
        output = input - tf.math.reduce_mean(
            input,
            axis=-2,
            keepdims=True
        )
        cov = tf.linalg.matmul(output, output, transpose_a=True)
        _, output = tf.linalg.eigh(cov)
        output = tf.transpose(output, perm=[0, 2, 1])
        return output

    def get_config(self):
        base_config = super().get_config()
        config = {
            "emb_dim": self.emb_dim,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(
    package="BinaryLoadings"
)
class BinaryLoadings(tf.keras.layers.Layer):
    def __init__(
        self,
        enc_layers,
        enc_heads,
        dff,
        dropout,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.enc_layers = enc_layers
        self.enc_heads = enc_heads
        self.dff = dff
        self.dropout = dropout
        self.encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=enc_layers,
            num_attention_heads=enc_heads,
            intermediate_size=dff,
            dropout_rate=dropout,
            norm_first=True,
            activation='relu')
        self.binary_ff = tf.keras.layers.Dense(3)

    def call(self, inputs, training=None):
        output = inputs
        output = self.encoder(inputs, training=training)
        return self.binary_ff(output)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "enc_layers": self.enc_layers,
            "enc_heads": self.enc_heads,
            "dff": self.dff,
            "dropout": self.dropout,
        }
        return {**base_config, **config}
    
@tf.keras.saving.register_keras_serializable(
    package="BinaryLoadings"
)
class Loadings(tf.keras.layers.Layer):
    def __init__(
        self,
        enc_layers,
        enc_heads,
        dff,
        dropout,
        output_ff,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.enc_layers = enc_layers
        self.enc_heads = enc_heads
        self.dff = dff
        self.dropout = dropout
        self.encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=enc_layers,
            num_attention_heads=enc_heads,
            intermediate_size=dff,
            dropout_rate=dropout,
            norm_first=True,
            activation='relu')
        self.ff = tf.keras.layers.Dense(output_ff)

    def call(self, inputs, training=None):
        output = inputs
        output = self.encoder(inputs, training=training)
        return self.ff(output)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "enc_layers": self.enc_layers,
            "enc_heads": self.enc_heads,
            "dff": self.dff,
            "dropout": self.dropout,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(
    package="ProjectDown"
)
class ProjectDown(tf.keras.layers.Layer):
    def __init__(
        self,
        emb_dim,
        dims,
        reduce_dim,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.dims = dims
        self.reduce_dim = reduce_dim
        if dims == 3:
            shape = [None, emb_dim, emb_dim]
        else:
            shape = [None, emb_dim]

        self.proj_down = tf.keras.layers.Dense(1)
        self.proj_down.build(shape)
        self.reduce_dim = reduce_dim

    def call(self, inputs):
        if self.reduce_dim:
            output = tf.squeeze(self.proj_down(inputs), axis=-1)
        else:
            output = self.proj_down(inputs)
        return output

    def get_config(self):
        base_config = super().get_config()
        config = {
            "emb_dim": self.emb_dim,
            "dims": self.dims,
            "reduce_dim": self.reduce_dim
        }
        return {**base_config, **config}
