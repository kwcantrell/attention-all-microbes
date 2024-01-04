import tensorflow as tf
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
        reduce_tensor=False,
        kernel_initializer="glorot_uniform",
        input_max_length=None,
        seq_axis=2,
        normalize_output=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dff = dff
        self.reduce_tensor = reduce_tensor
        self.kernel_initializer=kernel_initializer
        self.seq_axis = seq_axis
        self.input_max_length = input_max_length
        self.normalize_output = normalize_output

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
        self.dropout = tf.keras.layers.Dropout(.50)
        
        self.pos_emb_red = tfm.nlp.layers.PositionEmbedding(max_length=self.dff, seq_axis=seq_axis)

    def get_config(self):
        config = {
            "dff": self.dff,
            "reduce_tensor": self.reduce_tensor,
            "kernel_initializer": self.kernel_initializer,
            "input_max_length":self.input_max_length,
            "seq_axis":self.seq_axis,
            "normalize_out": self.normalize_output,
        }
        return config
    
    def call(self, inputs):
        if self.input_max_length:
            inputs += self.pos_emb_input(inputs)
        output = tf.keras.activations.relu(tf.einsum("...ij,jk->...kj", inputs, self.kernel_dff))

        if self.normalize_output:
            output = self.norm(output)
        output += self.pos_emb_red(output)

        if self.reduce_tensor:
            output = tf.reduce_sum(output, axis=2)
        
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
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(.50)
        self.asv_pos_emb = NucleotideEinsum(
            64,
            reduce_tensor=False,
        )
        self.asv_pos_emb2 = NucleotideEinsum(
            64,
            reduce_tensor=False,
            input_max_length=64,
            seq_axis=1,
            normalize_output=True,
        )
        self.asv_pos_emb3 = NucleotideEinsum(
            64,
            reduce_tensor=False,
            input_max_length=64,
            seq_axis=1,
            normalize_output=True,
        )
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout2 = tf.keras.layers.Dropout(.50)
        

        self.encoding_blocks = tfm.nlp.models.TransformerEncoder(
            num_layers=num_enc_layers,
            num_attention_heads=8,
            intermediate_size=2048,
            # dropout_rate=dropout,
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
        output = self.asv_pos_emb(input)
        output = self.asv_pos_emb2(output)
        output = self.asv_pos_emb3(output)
        return self.encoding_blocks(output)
    

@tf.keras.saving.register_keras_serializable(
    package="amplicon_gpt.layers"
)
class ReadHead(tf.keras.layers.Layer):
    def __init__(
            self,
            dff,
            **kwargs
        ):
        super().__init__(name='read_head', **kwargs)
        self.dff = dff
        self.norm = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(128, activation='gelu')
        self.dense2 = tf.keras.layers.Dense(32)
        self.asv_pos_emb = NucleotideEinsum(
            64,
            reduce_tensor=False,
            input_max_length=64,
            seq_axis=1,
            normalize_output=True,
        )
        self.asv_pos_emb2 = NucleotideEinsum(
            64,
            reduce_tensor=False,
            input_max_length=64,
            seq_axis=1,
            normalize_output=True,
        )

    def get_config(self):
        config = {
            "dff": self.dff,
        }
        return config
    
    def call(self, inputs):
        output = self.asv_pos_emb(inputs)
        output = self.asv_pos_emb2(output)
        output = tf.reduce_sum(output, axis=1)
        output = self.norm(output)
        output = self.dense(output)
        return self.dense2(output)