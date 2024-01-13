import os
from collections import namedtuple
import tensorflow as tf
from amplicon_gpt.initializers import UnitUniform
import tensorflow_models as tfm

@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layers"
)
class NucleotideEinsum(tf.keras.layers.Layer):
    """A layer that encodes an arbitrary tensor of nucleotide embeddings
    of size (..., N, E) and encodes it as a fixed size tensor of size
    (..., E, E*). Dimension N represents the total number of nucleotides,
    dimension E represents the embedding dimension of the nucleotides, and
    E* is an the intermediate dimension.

    Args:
      dff: The hidden dimension of the intermediate pointwise project

    Examples:
    >>> embeddings = tf.reshape(tf.range(0,2*6, dtype=tf.float32),(1,2,3,2))
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
        reduce_tensor=True,
        kernel_initializer="glorot_uniform",
        use_pos_emb=None,
        max_length=150,
        seq_axis=2,
        **kwargs
    ):
        super().__init__(name='nucleotide_einsum', **kwargs)
        self.dff = dff
        self.reduce_tensor = reduce_tensor
        self.kernel_initializer=kernel_initializer
        self.use_pos_emb = use_pos_emb
        self.seq_axis = seq_axis
        self.max_length = max_length
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(max_length=max_length, seq_axis=seq_axis)
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(.50)

        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout2 = tf.keras.layers.Dropout(.50)
    
    def build(self, input_shape):
        self.kernel_dff = self.add_weight(
            "kernel_dff",
            shape=(input_shape[-1], self.dff),
            initializer=tf.keras.initializers.get(self.kernel_initializer)
        )
        if self.reduce_tensor:
            self.kernel_emb = self.add_weight(
                "kernel_emb",
                shape=(input_shape[-1], self.dff),
                initializer=tf.keras.initializers.get(self.kernel_initializer)
            )

    def get_config(self):
        config = {
            "dff": self.dff,
            "reduce_tensor": self.reduce_tensor,
            "kernel_initializer": self.kernel_initializer,
            "use_pos_emb":self.use_pos_emb,
            "max_length":self.max_length,
            "seq_axis":self.seq_axis
        }
        return config
    
    def call(self, inputs):
        if self.use_pos_emb:
            inputs += self.pos_emb(inputs)

        output = self.norm(inputs)
        output = self.dropout(output)
        output = tf.keras.activations.relu(tf.einsum("...ij,jk->...jk", output, self.kernel_dff))

        if self.reduce_tensor:
            output = self.norm2(output)
            output = self.dropout2(output)
            output = tf.keras.activations.relu(tf.einsum("...ij,ij->...i", output, self.kernel_emb))
        
        return output
    
@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layers")
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
        self.asv_pos_emb = NucleotideEinsum(
            128,
            reduce_tensor=False,
            use_pos_emb=False)
        self.asv_pos_emb2 = NucleotideEinsum(
            64,
            reduce_tensor=False,
            use_pos_emb=False)
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(.50)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout2 = tf.keras.layers.Dropout(.50)

        self.encoding_blocks = tfm.nlp.models.TransformerEncoder(
            num_layers=num_enc_layers,
            num_attention_heads=8,
            intermediate_size=2048,
            dropout_rate=dropout,
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
        output = self.norm(input)
        output = self.dropout(output)
        output = self.asv_pos_emb(output)

        output = self.norm2(output)
        output = self.dropout2(output)
        output = self.asv_pos_emb2(output)

        return self.encoding_blocks(output)
    

@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layers"
)
class ReadHead(tf.keras.layers.Layer):
    def __init__(
            self,
            dff,
            kernel_initializer="glorot_uniform",
            **kwargs
        ):
        super().__init__(name='read_head', **kwargs)
        self.dff = dff
        self.kernel_initializer=kernel_initializer
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(.50)
    
    def build(self, input_shape):
        self.kernel_dff = self.add_weight(
            "kernel_dff",
            shape=(input_shape[1], self.dff),
            initializer=tf.keras.initializers.get(self.kernel_initializer)
        )

    def get_config(self):
        config = {
            "dff": self.dff,
            "kernel_initializer": self.kernel_initializer,
        }
        return config
    
    def call(self, inputs):
        output = self.norm(inputs)
        output = self.dropout(output)
        output = tf.keras.activations.relu(tf.einsum("...ij,ij->...j", output, self.kernel_dff))
        return output
    
            