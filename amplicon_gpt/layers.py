import tensorflow as tf
import tensorflow_models as tfm


@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layer"
)
class MultiHeadPCAProjection(tf.keras.layers.Layer):
    def build(self, input_shape):
        shape = [x if x is not None else -1 for x in input_shape]
        num_heads = 8
        emb_shape = shape[-1]
        head_size = emb_shape // num_heads
        reshape = shape[:-1] + [num_heads, head_size]

        first_transp = [i for i in range(len(reshape))]
        first_transp = first_transp[:-3] + [first_transp[-2],
                                            first_transp[-3],
                                            first_transp[-1]]
        second_transp = reshape[:-3] + [emb_shape]

        self.u_proj_vec = self.add_weight(
            "u_proj_vec",
            shape=[1, head_size],
            initializer='glorot_uniform',
            trainable=True
        )
        self.linear_transform = tf.keras.layers.Dense(emb_shape, 'relu')
        init_tup = (reshape,
                    tf.constant(emb_shape - 1, dtype=tf.float32),
                    first_transp, second_transp)
        self.compute_proj = MultiHeadPCAProjection.init_proj(*init_tup)
    
    def init_proj(reshape, emb_shape, first_transp, second_transp):
        @tf.function(reduce_retracing=True, jit_compile=True)
        def compute_proj(X, u):
            # create heads
            X = tf.reshape(X, shape=reshape)
            X = tf.transpose(X, perm=first_transp)
            
            # compute pca projections
            X_h = tf.reduce_mean(X, axis=-1, keepdims=True)
            X = tf.subtract(X, X_h)
            cov = tf.linalg.matmul(X, X, transpose_a=True)
            if not tf.is_symbolic_tensor(X):
                cov /= tf.constant(tf.shape(X)[-2] - 1, dtype=tf.float32)
            _, eig_vec = tf.linalg.eigh(cov)
            projs = tf.einsum('...ij,...j->...i', eig_vec, u)

            # join heads
            concat_projs = tf.einsum('...hi->...ih', projs)
            output = tf.reshape(concat_projs, shape=second_transp)
            return output
        return compute_proj

    def call(self, inputs):
        output = self.linear_transform(inputs)
        output = self.compute_proj(output, self.u_proj_vec)
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


@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layers"
)
class SampleEncoder(tf.keras.layers.Layer):
    def __init__(
            self,
            dropout,
            num_enc_layers,
            num_heads,
            dff,
            norm_first,
            **kwargs
    ):
        super().__init__(name="sample_encoder", **kwargs)
        self.dropout = dropout
        self.num_enc_layers = num_enc_layers
        self.num_heads = num_heads
        self.dff = dff
        self.norm_first = norm_first

        self.encoding_blocks = tfm.nlp.models.TransformerEncoder(
            num_layers=num_enc_layers,
            num_attention_heads=8,
            intermediate_size=2048,
            dropout_rate=0.5,
            norm_first=True,
            activation='gelu',
        )

    def get_config(self):
        config = {
            "dropout": self.dropout,
            "num_enc_layers": self.num_enc_layers,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "norm_first": self.norm_first,
        }
        return config

    def call(self, input, training=False):
        print(training)
        return self.encoding_blocks(input)
    

@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layers"
)
class ReadHead(tf.keras.layers.Layer):
    def __init__(
            self,
            dff,
            output_dim,
            **kwargs
    ):
        super().__init__(name='read_head', **kwargs)
        self.dff = dff
        self.output_dim = output_dim
        self.dense = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_dim)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.pca_proj = MultiHeadPCAProjection()
        self.linear_activation = tf.keras.layers.Activation('linear',
                                                            dtype='float32')

    def get_config(self):
        config = {
            "dff": self.dff,
            "output_dim": self.output_dim
        }
        return config

    def call(self, inputs):
        output = self.pca_proj(inputs)
        return output

