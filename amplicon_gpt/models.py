import tensorflow as tf
from amplicon_gpt.layers import NucleotideSequenceEmbedding, SampleEncoder, UniFracEncoder

# dropout=0.5
# batch_size=8

# @tf.keras.saving.register_keras_serializable()
# class UniFrac(tf.keras.Model):
#     def __init__(
#         self,
#         d_model,
#         dff,
#         num_heads,
#         num_enc_layers,
#         norm_first=False,
#         name="unifrac",
#         **kwargs
#     ):
#         """
#         need sequence tokenizer
#                 batch_size
#                 dropout
#         """
#         super().__init__(name=name, **kwargs)
#         self.d_model = d_model
#         self.dff = dff
#         self.num_heads = num_heads
#         self.num_enc_layers = num_enc_layers
        
#         self.seq_token = sequence_tokenizer
#         self.inputs = tf.keras.layers.Input(shape=(None, 1), batch_size=batch_size, dtype=tf.string, ragged=True)
#         self.embedding = tf.keras.layers.Embedding(5, d_model, input_length=100)
#         self.nuc = NucleotideSequenceEmbedding(d_model, dropout)
#         self.samp = SampleEncoder(d_model, dropout, num_enc_layers, num_heads, dff, norm_first)
#         self.lstm = tf.keras.layers.LSTM(d_model, dropout=dropout, return_sequences=True)
#         self.unif = UniFracEncoder(dff)

    
#     def call(self, inputs, training=None):
#         @tf.function(
#             input_signature=[tf.RaggedTensorSpec(
#                 tf.TensorShape([batch_size, None, 1]), dtype=tf.string, ragged_rank=1)],
#             reduce_retracing=True,
#             jit_compile=False
#         ) 
#         def _call(input):   
#             output = input.to_tensor()
#             output = self.seq_token(output)
#             mask = tf.reduce_any(
#                 tf.math.equal(output, tf.constant(0, dtype=tf.int64)),
#                 axis=2
#             )
            
#             output = self.embedding(output)
#             output = self.nuc(output)
#             output = self.samp(output, mask=mask)
#             output = self.lstm(output)
#             return self.unif(output)
#         return _call(inputs)