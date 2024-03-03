import tensorflow as tf
import tensorflow_models as tfm


@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layer"
)
class MultiHeadPCAProjection(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim,
                 dropout=0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def build(self, input_shape):
        shape = [x if x is not None else -1 for x in input_shape]
        num_heads = 4
        emb_shape = shape[-1]
        head_size = emb_shape // num_heads
        reshape = shape[:-1] + [num_heads, head_size]
        first_transp = [i for i in range(len(reshape))]
        first_transp = first_transp[:-3] + [first_transp[-2],
                                            first_transp[-3],
                                            first_transp[-1]]
        second_transp = [i for i in range(len(reshape))]
        second_transp = second_transp[:-3] + [second_transp[-2],
                                              second_transp[-3],
                                              second_transp[-1]]
        second_reshape = shape[:-2] + [emb_shape, self.hidden_dim]
        self.linear_transform = tf.keras.layers.Dense(emb_shape,
                                                      activation='relu')

        self.dff = tf.keras.layers.Dense(self.hidden_dim,
                                         activation='relu',
                                         use_bias=False)
        self.point = tf.keras.layers.Dense(1,
                                           activation='relu',
                                           use_bias=False)
        self.dropout = tf.keras.layers.Dropout(self.dropout)
        init_tup = (reshape,
                    first_transp,
                    second_transp,
                    second_reshape,
                    self.dff,
                    self.point)
        self.second = second_transp
        self.compute_proj = MultiHeadPCAProjection.init_proj(*init_tup)

    def init_proj(reshape,
                  first_transp,
                  second_transp,
                  second_reshape,
                  dff,
                  point):
        @tf.function(reduce_retracing=True, jit_compile=True)
        def compute_proj(X):
            ONE = tf.constant(1)
            X = tf.reshape(X, shape=reshape)
            X = tf.transpose(X, perm=first_transp)
            X -= tf.reduce_mean(X, axis=-2, keepdims=True)
            cov = tf.linalg.matmul(X, X, transpose_a=True)
            if not tf.is_symbolic_tensor(X):
                cov /= tf.constant(reshape[-ONE] - ONE,
                                   dtype=tf.float32)
            _, eig_vec = tf.linalg.eigh(cov)
            proj = dff(eig_vec)
            proj = tf.transpose(proj, perm=second_transp)
            proj = tf.reshape(proj, shape=second_reshape)
            proj = tf.squeeze(point(proj), axis=-1)
            return proj
        return compute_proj

    def call(self, inputs, training):
        output = self.linear_transform(inputs)
        output = self.compute_proj(output)
        return self.dropout(output, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout
        })
        return config


@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layers"
)
class ReadHead(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_dim,
            output_dim,
            dropout=0.,
            **kwargs
    ):
        super().__init__(name='read_head', **kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.pca_proj = MultiHeadPCAProjection(
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            name='read_head_project')
        self.dense = tf.keras.layers.Dense(self.output_dim,
                                           use_bias=False)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout
        })
        return config

    def call(self, inputs):
        output = self.pca_proj(inputs)
        output = self.dense(output)
        return output


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
        self.dropout = tf.keras.layers.Dropout(0.50)

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

        return output
