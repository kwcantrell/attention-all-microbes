import tensorflow as tf
import tensorflow_models as tfm


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


@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layer"
)
class PCAProjector(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 num_layers,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        for i in range(self.num_layers):
            setattr(self,
                    f'pca_layer_{i}',
                    MultiHeadPCAProjection(
                        self.hidden_dim,
                        self.num_heads,
                        name=f'layer-{i}'))
            setattr(self,
                    f'ff-{i}',
                    tf.keras.layers.Dense(self.hidden_dim,
                                          activation='relu'))
        self.point = tf.keras.layers.Dense(1,
                                           activation='relu')

    def build(self, input_shape):
        shape = [x if x is not None else -1 for x in input_shape]
        emb_shape = shape[-1]
        self.back_proj = tf.keras.layers.Dense(emb_shape,
                                               activation='relu')

    def call(self, inputs):
        outputs = inputs
        for i in range(self.num_layers):
            outputs = getattr(self,
                              f'pca_layer_{i}')(outputs)
            outputs = getattr(self,
                              f'ff-{i}')(outputs)
        outputs = tf.squeeze(self.point(outputs), axis=-1)
        outputs = self.back_proj(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers
        })
        return config


@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layer"
)
class MultiHeadPCAProjection(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.nuc_ein = NucleotideEinsum(self.hidden_dim,
                                        reduce_tensor=False)

    def build(self, input_shape):
        shape = [x if x is not None else -1 for x in input_shape]
        # emb_shape = shape[-1]
        # self.hidden_dim = 256
        # self.num_heads = 16
        self.dff2 = tf.keras.layers.Dense(256, activation='relu')
        self.linear_up_scale = tf.keras.layers.Dense(self.hidden_dim,
                                                     activation='relu')
        # occurs after up scaling
        head_size = self.hidden_dim // self.num_heads
        # reshape = shape[:-2] + [512,
        #                         self.num_heads,
        #                         head_size]
        reshape = shape[:-1] + [self.num_heads, head_size]
        first_transp = [i for i in range(len(reshape))]
        first_transp = first_transp[:-3] + [first_transp[-2],
                                            first_transp[-3],
                                            first_transp[-1]]
        eig_trasp = [i for i in range(len(reshape))]
        eig_trasp = eig_trasp[:-2] + [eig_trasp[-1], eig_trasp[-2]]
        second_transp = [i for i in range(len(reshape))]
        second_transp = second_transp[:-3] + [second_transp[-2],
                                              second_transp[-3],
                                              second_transp[-1]]
        dff_dim = self.hidden_dim
        second_reshape = shape[:-2] + [self.hidden_dim, dff_dim]
        self.dff = tf.keras.layers.Dense(dff_dim, activation='relu')
        self.compute_proj = MultiHeadPCAProjection.init_proj(reshape,
                                                             first_transp,
                                                             eig_trasp)
        self.reshape = reshape
        self.first_transp = first_transp
        self.second_transp = second_transp
        self.second_reshape = second_reshape

    def init_proj(reshape, first_transp, eig_trasp):
        @tf.function(jit_compile=True)
        def compute_proj(X):
            if not tf.is_symbolic_tensor(X):
                X = X - tf.math.reduce_mean(X, axis=-2, keepdims=True)
            cov = tf.linalg.matmul(X, X, transpose_a=True)
            _, eig_vec = tf.linalg.eigh(cov)
            return tf.transpose(eig_vec, perm=eig_trasp)
        return compute_proj

    def call(self, inputs):
        output = self.dff2(inputs)
        output = self.linear_up_scale(output)
        # if not tf.is_symbolic_tensor(output):
        #     self.reshape[1] = tf.shape(inputs)[1]
        output = tf.reshape(output, shape=self.reshape)
        output = tf.transpose(output, perm=self.first_transp)
        output = self.compute_proj(output)
        output = self.dff(output)
        output = tf.transpose(output, perm=self.second_transp)
        output = tf.reshape(output, shape=self.second_reshape)
        # output = self.nuc_ein(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
        })
        return config


@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layers"
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
        super().__init__(name='read_head', **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        # self.pca_proj = PCAProjector(
        #     hidden_dim=self.hidden_dim,
        #     num_heads=self.num_heads,
        #     num_layers=self.num_layers,
        #     name='read_head_project')
        self.pca_proj = NucleotideEinsum(64,
                                         reduce_tensor=True,
                                         normalize_output=True,
                                         seq_axis=1)
        self.dff = tf.keras.layers.Dense(128, activation='relu', use_bias=True)
        self.dense = tf.keras.layers.Dense(self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
        })
        return config

    def call(self, inputs):
        output = self.pca_proj(inputs)
        output = self.dff(output)
        output = self.dense(output)
        return output
