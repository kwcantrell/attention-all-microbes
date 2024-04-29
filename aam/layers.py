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
            5,
            pca_hidden_dim,
            embeddings_initializer="ones",
            input_length=max_bp,
        )
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(
                max_length=max_bp,
                seq_axis=2
        )
        self.pca_layer = PCAProjector(
            hidden_dim=pca_hidden_dim,
            num_heads=8,
            num_layers=pca_layers,
            dropout=dropout_rate
        )

        self.attention_layer = tfm.nlp.models.TransformerEncoder(
            num_layers=6,
            num_attention_heads=2,
            intermediate_size=1024,
            dropout_rate=dropout_rate,
            norm_first=True,
            activation='relu',
        )

    def call(self, inputs, training=None):
        seq, rclr = tf.nest.flatten(inputs, expand_composites=True)
        output = self.emb_layer(seq)
        output = output + self.pos_emb(output)
        output = self.pca_layer(output)
        output = tf.multiply(
            output,
            tf.expand_dims(rclr, axis=-1)
        )
        output = self.attention_layer(output, training=training)
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
    package="amplicon_gpt.layers"
)
class NucleotideEinsum(tf.keras.layers.Layer):
    """A layer that encodes an arbitrary tensor of nucleotide embeddings
    of size (..., N, E) and encodes it as a fixed-size tensor of size
    (..., E, E*). Dimension N represents the total number of nucleotides,
    dimension E represents the embedding dimension of the nucleotides, and
    E* is an intermediate dimension.

    Args:
        dff: The hidden dimension of the intermediate pointwise project

    Examples:
        >>> embeddings = tf.reshape(
                tf.range(0, 2 * 6, dtype=tf.float32),
                (1, 2, 3, 2))
        >>> embedding
        <tf.Tensor: shape=(1, 2, 3, 2), dtype=float32, numpy=
        array([[[[ 0.,  1.],
                 [ 2.,  3.],
                 [ 4.,  5.]],
                [[ 6.,  7.],
                 [ 8.,  9.],
                 [10., 11.]]]], dtype=float32)>

        >>> einsum_dense = NucleotideEinsum(dff=8, kernel_initializer="ones")
        >>> einsum_dense(embeddings)
        <tf.Tensor: shape=(1, 2, 2), dtype=float32, numpy=
        array([[[ 48.,  72.],
                [192., 216.]]], dtype=float32)>
    """
    def __init__(
        self,
        dff,
        reduce_tensor=False,
        kernel_initializer="glorot_uniform",
        input_max_length=None,
        seq_axis=2,
        normalize_output=False,
        activation='relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dff = dff
        self.reduce_tensor = reduce_tensor
        self.kernel_initializer = kernel_initializer
        self.seq_axis = seq_axis
        self.input_max_length = input_max_length
        self.normalize_output = normalize_output
        self.activation = activation
        self.activation_function = tf.keras.activations.get(activation)

    def build(self, input_shape):
        seq_axis = len(input_shape) - 2
        if self.input_max_length:
            self.pos_emb_input = tfm.nlp.layers.PositionEmbedding(
                max_length=self.input_max_length,
                seq_axis=seq_axis
            )

        self.kernel_dff = self.add_weight(
            "kernel_dff",
            shape=(input_shape[-1], self.dff),
            initializer=tf.keras.initializers.get(self.kernel_initializer)
        )
        self.norm = tf.keras.layers.LayerNormalization()

        self.pos_emb_red = tfm.nlp.layers.PositionEmbedding(
                max_length=self.dff, seq_axis=seq_axis)

    def get_config(self):
        config = {
            "dff": self.dff,
            "reduce_tensor": self.reduce_tensor,
            "kernel_initializer": self.kernel_initializer,
            "input_max_length": self.input_max_length,
            "seq_axis": self.seq_axis,
            "normalize_out": self.normalize_output,
        }
        return config

    def call(self, inputs):
        if self.input_max_length:
            inputs += self.pos_emb_input(inputs)
        output = self.activation_function(tf.einsum("...ij,jk->...kj",
                                          inputs, self.kernel_dff))

        output += self.pos_emb_red(output)

        if self.reduce_tensor:
            output = tf.reduce_sum(output, axis=2)

        return self.norm(output)


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
        self.point = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = inputs
        outputs = self.pca_layer(outputs)
        outputs = tf.squeeze(self.point(outputs), axis=-1)
        return outputs
    
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
        self.norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        shape = [x if x is not None else -1 for x in input_shape]
        emb_shape = shape[-1]
        self.linear_up_scale = tf.keras.layers.Dense(self.hidden_dim,
                                                     activation='relu',
                                                     use_bias=False)
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
        self.dff = tf.keras.layers.Dense(head_size,
                                         activation='relu',
                                         use_bias=False)
        self.dropout = tf.keras.layers.Dropout(self.dropout)
        init_tup = (reshape,
                    first_transp,
                    second_transp,
                    second_reshape,
                    self.dff)
        self.second = second_transp
        self.compute_proj = MultiHeadPCAProjection.init_proj(*init_tup)

    def init_proj(reshape,
                  first_transp,
                  second_transp,
                  second_reshape,
                  dff):
        @tf.function(jit_compile=True)
        def compute_proj(X):
            X = tf.reshape(X, shape=reshape)
            X = tf.transpose(X, perm=first_transp)
            # X -= tf.reduce_mean(X, axis=-1, keepdims=True)
            cov = tf.linalg.matmul(X, X, transpose_a=True)
            eig_values, eig_vec = tf.linalg.eigh(cov)
            proj = tf.transpose(
                tf.matmul(
                    tf.linalg.diag(
                        eig_values
                    ),
                    eig_vec,
                ),
                perm=second_transp
            )
            proj = tf.reshape(proj, shape=second_reshape)
            return proj
        return compute_proj

    def call(self, inputs):
        output = self.linear_up_scale(inputs)
        output = self.compute_proj(output)
        # output = self.norm(output)
        return self.dropout(output)

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
            tf.keras.layers.Dense(self.output_dim),
        ])

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
        output = inputs[:, -1, :]
        output = self.read_head(output, training=training)
        return output
