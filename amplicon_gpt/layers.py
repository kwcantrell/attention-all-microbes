import os
from collections import namedtuple
import tensorflow as tf
import keras_nlp
from amplicon_gpt.initializers import UnitUniform

@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="NucleotideSequenceEmbedding")
class NucleotideSequenceEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, dropout, **kwargs):
        super().__init__(name="nucleotide_sequence_embedding", **kwargs)
        self.embedding_dim = embedding_dim
        self.lstm = tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(embedding_dim, dropout=dropout, return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='gelu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(1, activation='gelu'),
            tf.keras.layers.Flatten()
        ])

    def build(self, input_shape):
        dense_shape = input_shape[0], input_shape[2], input_shape[3]
        def vec_layer(input):
            output = tf.transpose(input, perm=[0,2,1])
            return self.dense(output)
        
        self.vectorize_layer = tf.function(
            vec_layer,
            input_signature=[tf.TensorSpec((dense_shape), dtype=tf.float32)],
            jit_compile=True
        )
    
    def call(self, input, mask=None, training=None):
        output = self.lstm(input)
        output = self.dropout(output)
        output = tf.transpose(output, perm=[1,0,2,3])
        output = tf.vectorized_map(self.vectorize_layer, output, fallback_to_while_loop=False)
        output = tf.transpose(output, perm=[1,0,2])
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
                "embedding_dim": self.embedding_dim,
                "dropout": self.dropout
        })
        return config
        
@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="PositionEncoder")
class SampleEncoder(tf.keras.layers.Layer):
    def __init__(self, nucleotide_embedding_dim, dropout, num_enc_layers, num_heads, dff, norm_first, **kwargs):
        super().__init__(name="sample_encoder", **kwargs)
        self.nucleotide_embedding_dim = nucleotide_embedding_dim
        self.asv_pos_emb = tf.keras.layers.LSTM(nucleotide_embedding_dim, dropout=dropout, return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.encoding_blocks = [
            keras_nlp.layers.TransformerEncoder(num_heads=num_heads, dropout=dropout,
                    activation='gelu', intermediate_dim=dff, normalize_first=norm_first,
                    name=f'base_encoder_block_{i}')
            for i in range(num_enc_layers)]
    
    def build(self, input_shape):
        @tf.function(
                jit_compile=True
        )
        def pad(x):
            padddings = tf.constant([[0, 128], [0,0]])
            output = tf.pad(x, padddings)
            output = tf.strided_slice(output, begin=[0,0], end=[128, input_shape[2]])
            return output
        self.pad = pad.get_concrete_function(tf.TensorSpec(shape=input_shape[1:], dtype=tf.float32))

        @tf.function(
                jit_compile=True
        )
        def run_layer(x):
            for layer in self.encoding_blocks:
                x = layer(x)
            return x
        self.run_layer = run_layer.get_concrete_function(
                tf.TensorSpec(shape=[input_shape[0], 128, input_shape[2]], dtype=tf.float32)
        )        


    def call(self, input, mask=None, training=None):
        output = self.asv_pos_emb(input)
        output = self.dropout(output)
        output = tf.vectorized_map(self.pad, output, fallback_to_while_loop=False)
        output = self.run_layer(output)
        return output
            
    def get_config(self):
        config = super().get_config()
        config.update({
                "dropout": self.dropout
        })
        return config
    
@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="UniFracEncoder")
class UniFracEncoder(tf.keras.layers.Layer):
    def __init__(self, dff, **kwargs):
        super().__init__(name="unifrac_embedding", **kwargs)
        # self.norm = tf.keras.layers.LayerNormalization()
        # self.ff = tf.keras.layers.Dense(64, activation='gelu')
        # self.dropout = tf.keras.layers.Dropout(0.05)
        self.dense = tf.keras.layers.Dense(1)
    
    def build(self, input_shape):
        def vec_layer(input):
            output = self.ff(input)
            output = self.norm(output)
            return tf.squeeze(output)
        
        self.vectorize_layer = tf.function(
            vec_layer,
            input_signature=[tf.TensorSpec((1, input_shape[2]), dtype=tf.float32)],
            jit_compile=True
        )
    def call(self, input, training=None):
        # output = tf.math.reduce_sum(input, axis=1)
        # output = tf.expand_dims(output, axis=1)
        # output = tf.vectorized_map(self.vectorize_layer, output, fallback_to_while_loop=False)
        # output = self.dropout(output)
        return tf.squeeze(self.dense(input))

            