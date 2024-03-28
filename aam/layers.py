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
        self.point = tf.keras.layers.Dense(1)

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
        self.norm = tf.keras.layers.LayerNormalization(axis=-2)

    def build(self, input_shape):
        shape = [x if x is not None else -1 for x in input_shape]
        self.linear_up_scale = tf.keras.layers.Dense(self.hidden_dim)
        # occurs after up scaling
        head_size = self.hidden_dim // self.num_heads
        self.dff = tf.keras.layers.Dense(head_size)

        reshape = shape[:-1] + [self.num_heads, head_size]
        first_transp = [i for i in range(len(reshape))]
        first_transp = first_transp[:-3] + [first_transp[-2],
                                            first_transp[-1],
                                            first_transp[-3]]
        second_transp = [i for i in range(len(reshape))]
        second_transp = second_transp[:-3] + [second_transp[-2],
                                              second_transp[-3],
                                              second_transp[-1]]
        second_reshape = shape[:-2] + [head_size, self.hidden_dim]
        eig_trasp = [i for i in range(len(reshape))]
        eig_trasp = eig_trasp[:-2] + [eig_trasp[-1], eig_trasp[-2]]

        self.compute_proj = MultiHeadPCAProjection.init_proj(reshape,
                                                             first_transp,
                                                             eig_trasp,
                                                             second_transp,
                                                             second_reshape,
                                                             self.dff)

    def init_proj(reshape,
                  first_transp,
                  eig_trasp,
                  second_transp,
                  second_reshape,
                  dff):
        @tf.function(jit_compile=True)
        def compute_proj(X):
            if not tf.is_symbolic_tensor(X):
                reshape[1] = tf.shape(X)[1]
            output = tf.reshape(X, shape=reshape)
            output = tf.transpose(output, perm=first_transp)
            if not tf.is_symbolic_tensor(X):
                output = output - tf.math.reduce_mean(output,
                                                      axis=-1,
                                                      keepdims=True)
            cov = tf.linalg.matmul(output, output, transpose_b=True)
            _, output = tf.linalg.eigh(cov)
            output = tf.transpose(output, perm=eig_trasp)
            output = dff(output)
            output = tf.transpose(output, perm=second_transp)
            output = tf.reshape(output, shape=second_reshape)
            return output
        return compute_proj

    def call(self, inputs):
        if not tf.is_symbolic_tensor(inputs):
            inputs = self.norm(inputs)
        output = self.linear_up_scale(inputs)
        output = self.compute_proj(output)
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
            dropout=0.0,
            **kwargs
    ):
        super().__init__(name='read_head', **kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.norm = tf.keras.layers.LayerNormalization(axis=-2)
        self.pca_proj = NucleotideEinsum(128,
                                         reduce_tensor=True,
                                         normalize_output=True,
                                         seq_axis=1)
        self.dff = tf.keras.layers.Dense(128,
                                         activation='relu',
                                         use_bias=True)
        self.dense = tf.keras.layers.Dense(self.output_dim)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
        })
        return config

    def call(self, inputs, training=None):
        if not tf.is_symbolic_tensor(inputs):
            inputs = self.norm(inputs)
        output = self.pca_proj(inputs)
        output = self.dff(output)
        output = self.dropout(output, training=training)
        output = self.dense(output)
        return output