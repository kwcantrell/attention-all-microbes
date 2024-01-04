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
        self.dropout = dropout
        self.lstm = tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(embedding_dim, dropout=dropout, return_sequences=True))
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='gelu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout),
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
        self.dropout = dropout
        self.asv_pos_emb = keras_nlp.layers.PositionEmbedding(sequence_length=1600)
        self.encoding_blocks = [
            keras_nlp.layers.TransformerEncoder(num_heads=num_heads, dropout=dropout,
                    activation='gelu', intermediate_dim=dff, normalize_first=norm_first,
                    name=f'base_encoder_block_{i}')
            for i in range(num_enc_layers)]
    
    def call(self, input, mask=None, training=None):
        padding_mask=tf.cast(mask, dtype=tf.float32)
        microbime_pos = self.asv_pos_emb(input)
        output = tf.math.add(input, microbime_pos)
        
        for layer in self.encoding_blocks:
            output = layer(output, padding_mask=padding_mask)
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
        self.norm = tf.keras.layers.LayerNormalization()
        self.ff = tf.keras.layers.Dense(64, activation='gelu')
        self.dropout = tf.keras.layers.Dropout(0.05)
        self.dense = tf.keras.layers.Dense(32)
    
    def build(self, input_shape):
        # dense_shape = [1] + input_shape[2]
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
        output = tf.math.reduce_sum(input, axis=1)
        output = tf.expand_dims(output, axis=1)
        output = tf.vectorized_map(self.vectorize_layer, output, fallback_to_while_loop=False)
        output = self.dropout(output)
        return self.dense(output)

            