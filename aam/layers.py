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
    def __init__(
        self,
        hidden_dim,
        num_heads,
        num_layers,
        **kwargs
    ):
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
        self.point = tf.keras.layers.Dense(1)

    # def build(self, input_shape):
    #     shape = [x if x is not None else -1 for x in input_shape]
    #     emb_shape = shape[-1]
        # self.back_proj = tf.keras.layers.Dense(
        #     emb_shape,
        #     activation='relu'
        # )

    def call(self, inputs):
        outputs = inputs
        for i in range(self.num_layers):
            outputs = getattr(self,
                              f'pca_layer_{i}')(outputs)

        outputs = tf.squeeze(self.point(outputs), axis=-1)
        # outputs = self.back_proj(outputs)
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
    def __init__(
        self,
        hidden_dim,
        num_heads,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # self.norm = tf.keras.layers.LayerNormalization(axis=-2)

    def build(self, input_shape):
        shape = [x if x is not None else -1 for x in input_shape]
        # self.linear_up_scale = tf.keras.layers.Dense(self.hidden_dim)
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
            eig_vals, eig_vecs = tf.linalg.eigh(cov)
            output = tf.transpose(
                tf.matmul(
                    tf.linalg.diag(
                        eig_vals
                    ),
                    eig_vecs
                ),
                perm=eig_trasp
            )
            output = dff(output)
            output = tf.transpose(output, perm=second_transp)
            output = tf.reshape(output, shape=second_reshape)
            return output
        return compute_proj

    def call(self, inputs):
        if not tf.is_symbolic_tensor(inputs):
            inputs = self.norm(inputs)
        output = self.linear_up_scale(inputs)
        tf.print(tf.shape(output))
        output = self.compute_proj(inputs)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
        })
        return config

# @tf.keras.saving.register_keras_serializable(
#     package="amplicon_gpt.layer"
# )
# class MultiHeadPCAProjection(tf.keras.layers.Layer):
#     def __init__(self,
#                  hidden_dim,
#                  num_heads,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.norm = tf.keras.layers.LayerNormalization(axis=-2)

#     def build(self, input_shape):
#         shape = [x if x is not None else -1 for x in input_shape]
#         self.linear_up_scale = tf.keras.layers.Dense(self.hidden_dim)
#         # occurs after up scaling
#         head_size = self.hidden_dim // self.num_heads
#         self.dff = tf.keras.layers.Dense(head_size)

#         reshape = shape[:-1] + [self.num_heads, head_size]
#         first_transp = [i for i in range(len(reshape))]
#         first_transp = first_transp[:-3] + [first_transp[-2],
#                                             first_transp[-1],
#                                             first_transp[-3]]
#         second_transp = [i for i in range(len(reshape))]
#         second_transp = second_transp[:-3] + [second_transp[-2],
#                                               second_transp[-3],
#                                               second_transp[-1]]
#         second_reshape = shape[:-2] + [head_size, self.hidden_dim]
#         eig_trasp = [i for i in range(len(reshape))]
#         eig_trasp = eig_trasp[:-2] + [eig_trasp[-1], eig_trasp[-2]]

#         self.compute_proj = MultiHeadPCAProjection.init_proj(reshape,
#                                                              first_transp,
#                                                              eig_trasp,
#                                                              second_transp,
#                                                              second_reshape,
#                                                              self.dff)

#     def init_proj(reshape,
#                   first_transp,
#                   eig_trasp,
#                   second_transp,
#                   second_reshape,
#                   dff):
#         @tf.function(jit_compile=True)
#         def compute_proj(X):
#             if not tf.is_symbolic_tensor(X):
#                 reshape[1] = tf.shape(X)[1]
#             output = tf.reshape(X, shape=reshape)
#             output = tf.transpose(output, perm=first_transp)
#             if not tf.is_symbolic_tensor(X):
#                 output = output - tf.math.reduce_mean(output,
#                                                       axis=-1,
#                                                       keepdims=True)
#             cov = tf.linalg.matmul(output, output, transpose_b=True)
#             _, output = tf.linalg.eigh(cov)
#             output = tf.transpose(output, perm=eig_trasp)
#             output = dff(output)
#             output = tf.transpose(output, perm=second_transp)
#             output = tf.reshape(output, shape=second_reshape)
#             return output
#         return compute_proj

#     def call(self, inputs):
#         if not tf.is_symbolic_tensor(inputs):
#             inputs = self.norm(inputs)
#         output = self.linear_up_scale(inputs)
#         output = self.compute_proj(output)
#         return output

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "hidden_dim": self.hidden_dim,
#             "num_heads": self.num_heads,
#         })
#         return config


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
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.read_head = tf.keras.Sequential([
            NucleotideEinsum(
                128,
                reduce_tensor=False,
                normalize_output=True,
                seq_axis=1
            ),
            tf.keras.layers.Dense(
                32,
                activation='relu',
                use_bias=True
            ),
            tf.keras.layers.Dense(self.output_dim),
            tf.keras.layers.LayerNormalization(),
            # ProjectDown(32),
            tf.keras.layers.Dense(
                output_dim,
                use_bias=True
            )
        ])

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
        output = self.read_head(inputs)
        return output


# @tf.keras.saving.register_keras_serializable(
#     package="PCA"
# )
# class PCA(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.layer_norm = tf.keras.layers.LayerNormalization()

#     def build(self, input_shape):
#         self.perm = [i for i in range(len(input_shape))]
#         self.perm = self.perm[:-2] + [self.perm[-1], self.perm[-2]]

#     def call(self, inputs):
#         output = inputs - tf.reduce_mean(
#             tf.identity(inputs),
#             axis=1,
#             keepdims=True
#         )
#         eigen_values, eigen_vectors = tf.linalg.eigh(
#             tf.matmul(
#                 output,
#                 output,
#                 transpose_a=True
#             )
#         )
#         pca_transform = tf.transpose(
#             tf.matmul(
#                 tf.linalg.diag(
#                     eigen_values
#                 ),
#                 eigen_vectors
#             ),
#             perm=self.perm
#         )
#         return self.layer_norm(pca_transform)


# # @tf.keras.saving.register_keras_serializable(
# #     package="ProjectDown"
# # )
# # class ProjectDown(tf.keras.layers.Layer):
# #     def __init__(self, emb_dim, **kwargs):
# #         super().__init__(**kwargs)
# #         self.emb_dim = emb_dim
# #         self.ff = tf.keras.layers.Dense(
# #             emb_dim,
# #             activation='relu',
# #             use_bias=True
# #         )
# #         self.proj_down = tf.keras.layers.Dense(1)

# #     def call(self, inputs):
# #         outputs = self.ff(inputs)
# #         output = tf.squeeze(
# #             self.proj_down(outputs),
# #             axis=-1
# #         )
# #         return output

# #     def get_config(self):
# #         base_config = super().get_config()
# #         config = {
# #             "emb_dim": self.emb_dim,
# #         }
# #         return {**base_config, **config}


# @tf.keras.saving.register_keras_serializable(
#     package="NucleotideEmbedding"
# )
# class NucleotideEmbedding(tf.keras.layers.Layer):
#     def __init__(
#         self,
#         max_bp,
#         emb_dim,
#         d_model,
#         pca_hidden_dim,
#         pca_heads,
#         pca_layers,
#         dropout_rate,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.max_bp = max_bp
#         self.emb_dim = emb_dim
#         self.d_model = d_model
#         self.pca_hidden_dim = pca_hidden_dim
#         self.pca_heads = pca_heads
#         self.pca_layers = pca_layers
#         self.dropout_rate = dropout_rate

#         self.embedding_layer = tf.keras.layers.Embedding(
#             7,
#             emb_dim,
#             input_length=max_bp,
#             embeddings_initializer="ones",
#             name="embedding"
#         )
#         self.ff = tf.keras.layers.Dense(
#             256,
#             activation='relu',
#         )
#         self.norm = tf.keras.layers.LayerNormalization()
#         self.dropout = tf.keras.layers.Dropout(dropout_rate)
#         self.pos_embedding_layer = tfm.nlp.layers.PositionEmbedding(
#             max_length=max_bp,
#             seq_axis=2
#         )

#         def _component_block():
#             return (
#                 tf.keras.Sequential(
#                     [
#                         tf.keras.layers.Dense(
#                             128,
#                             activation='relu',
#                         ),
#                         tf.keras.layers.Dense(32),
#                         tf.keras.layers.LayerNormalization(),
#                     ]
#                 )
#             )
#         self.ff_pca = _component_block()
#         self.pca_projector =  tf.keras.Sequential([
#             PCA(),
#             ProjectDown(32),
#             tf.keras.layers.Dense(
#                 64,
#                 activation='relu'
#             ),
#             tf.keras.layers.LayerNormalization(),
#         ])

#     @tf.function(reduce_retracing=True, jit_compile=True)
#     def _inner(self, tensor):
#         output = tensor + self.pos_embedding_layer(tensor)
#         output = self.ff_pca(output)
#         output = self.pca_projector(output)
#         return output

#     def call(self, inputs, training=True):
#         asvs, clr = inputs
#         output = self.embedding_layer(asvs)
#         output = tf.math.multiply(
#             output,
#             tf.expand_dims(tf.expand_dims(clr, axis=-1), axis=-1)
#         )
#         output = self.norm(output)
#         output = self.dropout(output)
#         output = self.ff(output)
#         output = self._inner(output)
#         return output

#     def get_config(self):
#         base_config = super().get_config()
#         config = {
#             "max_bp:": self.max_bp,
#             "emb_dim": self.emb_dim,
#             "d_model:": self.d_model,
#             "pca_hidden_dim:": self.pca_hidden_dim,
#             "pca_heads:": self.pca_heads,
#             "pca_layers:": self.pca_layers,
#             "dropout_rate:": self.dropout_rate,
#         }
#         return {**base_config, **config}
