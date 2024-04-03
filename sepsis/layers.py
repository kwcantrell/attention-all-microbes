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
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=len(emb_vocab)+1,
            output_dim=token_dim,
            embeddings_initializer="ones"
        )
        self.d_model_ff = tf.keras.layers.Dense(
            d_model,
            activation='relu',
            use_bias=True
        )

        self.ff_pca = tf.keras.layers.Dense(ff_dim)
        self.ff_loadings = tf.keras.layers.Dense(ff_dim)

        self.flag = self.add_weight(trainable=False)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.1)

    def build(self, input_shape):
        self.d_model_ff.build([None, None, self.token_dim])
        self.ff_pca.build([None, None, self.d_model])
        self.ff_loadings.build([None, None, self.d_model])

    def _random_tokens(self, inputs):
        tokens, features_to_add, feature_size = inputs
        features_to_add = tf.squeeze(features_to_add)
        # get random tokens
        reduced_tokens = tokens[tokens > 0]
        reduced_tokens = tf.random.shuffle(reduced_tokens)
        tokens_selected = tf.minimum(feature_size, features_to_add)
        reduced_tokens = reduced_tokens[:tokens_selected]
        reduced_size = tf.cast(tf.shape(reduced_tokens)[0], dtype=tf.int64)
        reduced_tokens = tf.pad(
            reduced_tokens,
            [[feature_size - reduced_size, 0]]
        )
        return reduced_tokens

    def call(self, inputs, training=True):
        feature, rclr = inputs

        feature_tokens = self.feature_tokens(feature)
        
        # extend feature tensor to add random tokens
        feature_count = tf.reduce_sum(
            tf.cast(
                feature_tokens > 0,
                dtype=tf.float32
            ),
            axis=-1,
            keepdims=True,
        )        
        feature_count = tf.cast(
            feature_count * self.features_add_rate,
            dtype=tf.int64
        )

        batch_size = tf.cast(tf.shape(feature_tokens)[0], dtype=tf.int64)
        feature_size = tf.cast(tf.shape(feature_tokens)[1], dtype=tf.int64)
        feature_mask = tf.not_equal(feature_tokens, 0)

        batch_tokens = tf.expand_dims(self.tokens, axis=0)
        batch_tokens = tf.repeat(batch_tokens, batch_size, axis=0)
        token_indices = tf.equal(
            tf.expand_dims(feature_tokens, axis=-1),
            tf.expand_dims(batch_tokens, axis=1),
        )
        token_indices = tf.cast(token_indices, dtype=tf.int64)
        token_indices = tf.reduce_sum(
            token_indices,
            axis=1
        )
        not_in_mask = tf.equal(token_indices, 0)
        not_in_mask = tf.cast(not_in_mask, dtype=tf.int64)
        batch_tokens *= not_in_mask
        shuffled_tokens = tf.map_fn(
            self._random_tokens,
            (
                batch_tokens,
                feature_count,
                tf.repeat(feature_size, batch_size)
            ),
            fn_output_signature=tf.int64
        )
        shuffled_tokens = tf.random.shuffle(shuffled_tokens)

        # scatter current tokes to create mask for global
        randomized_tokens = feature_tokens + shuffled_tokens
        shuffled_tokens = tf.cast(shuffled_tokens, dtype=tf.float32)

        shuffled_mask = tf.cast(
            tf.greater(shuffled_tokens, 0),
            dtype=tf.float32
        )

        # get embedding vector
        feature_embedding = self.embedding_layer(randomized_tokens)
        output_embeddings = self.d_model_ff(feature_embedding)

        # scale by rclr
        randomized_rclr = rclr + shuffled_mask
        randomized_rclr = tf.expand_dims(randomized_rclr, axis=-1)
        output_embeddings = output_embeddings * randomized_rclr

        # prep embeddings for objective functions
        output_embeddings = self.layer_norm(output_embeddings)
        output_embeddings = self.dropout(output_embeddings, training=training)
        output_regression = self.ff_pca(output_embeddings)
        output_embeddings = self.ff_loadings(output_embeddings)

        return [
            feature_mask,
            randomized_tokens,
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


# @tf.keras.saving.register_keras_serializable(
#     package="ProjectDown"
# )
# class ProjectDown(tf.keras.layers.Layer):
#     def __init__(
#         self,
#         emb_dim,
#         dims,
#         reduce_dim,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.emb_dim = emb_dim
#         self.dims = dims
#         self.reduce_dim = reduce_dim
#         if dims == 3:
#             shape = [None, emb_dim, emb_dim]
#         else:
#             shape = [None, emb_dim]

#         self.proj_down = tf.keras.layers.Dense(1)
#         self.proj_down.build(shape)
#         self.reduce_dim = reduce_dim

#     def call(self, inputs):
#         if self.reduce_dim:
#             output = tf.squeeze(self.proj_down(inputs), axis=-1)
#         else:
#             output = self.proj_down(inputs)
#         return output

#     def get_config(self):
#         base_config = super().get_config()
#         config = {
#             "emb_dim": self.emb_dim,
#             "dims": self.dims,
#             "reduce_dim": self.reduce_dim
#         }
#         return {**base_config, **config}
