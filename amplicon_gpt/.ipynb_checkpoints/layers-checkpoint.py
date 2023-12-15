import os
import tensorflow as tf
import keras_nlp
from amplicon_gpt.initializers import UnitUniform

nucleotide_embedding_dim=256
nuc_norm_epsilon=1e-5
d_model = 128
dff = 512
num_heads = 6
num_enc_layers = 4
lstm_nuc_out = 128
lstm_seq_out = 128
emb_vec = 32
norm_first = False
conv_1_filter = 256
conv_2_filter = 64

@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="NucleotideSequenceEmbedding")
class NucleotideSequenceEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, dropout, **kwargs):
        super().__init__(name="nucleotide_sequence_embedding", **kwargs)
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        
        self.embedding = tf.keras.layers.Embedding(5, embedding_dim, input_length=150, mask_zero=False)
        # self.pos_embedding = keras_nlp.layers.PositionEmbedding(sequence_length=150)
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True, dropout=dropout)

    # def _pass_through_lstm(self, sequence, sequence_mask, training):
    #     emb = self.embedding(sequence)
    #     output = emb + self.pos_embedding(emb)
    #     output = tf.multiply(self.lstm(output, training=training), tf.expand_dims(sequence_mask, axis=-1))
    #     indicies = tf.multiply(tf.ones_like(sequence_mask), tf.range(0, 150, dtype=tf.float32))
    #     indicies = tf.multiply(indicies, sequence_mask)
    #     indicies = tf.argmax(indicies, axis=1)
    #     return tf.gather(output, indicies, batch_dims=1)
    
    def call(self, input, training=False):
        sequence_masks = tf.cast(tf.not_equal(input, 0), tf.float32)
        indicies = tf.multiply(tf.ones_like(sequence_masks), tf.range(0, 150, dtype=tf.float32))
        indicies = tf.multiply(indicies, sequence_masks)
        indicies = tf.argmax(indicies, axis=2)

        @tf.function
        def map_lstm(inputs):
            input, indicies = inputs
            output = self.lstm(input)
            return tf.gather(output, indices=[0], batch_dims=1)

        output = self.embedding(input)
        # pos_emb = self.pos_embedding(output)
        # output = tf.add(output, pos_emb)
        output = tf.map_fn(map_lstm, (output, indicies), fn_output_signature=tf.float32)
        # sequence_masks = tf.unstack(sequence_masks, axis=0)
        # output = tf.unstack(input, axis=0)
        # output = tf.nest.map_structure(lambda x, m: self._pass_through_lstm(x, m, training), output, sequence_masks, check_types=False)
        # output = tf.stack(output, axis=0)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
                "embedding_dim": self.embedding_dim,
                "dropout": self.dropout
        })
        return config
    
@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="ASVEncoder")
class ASVEncoder(tf.keras.layers.Layer):
    def __init__(self, dropout, **kwargs):
        super().__init__(name="asv_sequence_embedding", **kwargs)
        self.dropout = dropout
        self.asv_pos_emb = keras_nlp.layers.PositionEmbedding(sequence_length=1600)
        self.asv_norm = tf.keras.layers.LayerNormalization(epsilon=nuc_norm_epsilon)
        self.add = tf.keras.layers.Add()
        self.asv_dropout = tf.keras.layers.Dropout(dropout)
        self.encoder_blocks = [
            keras_nlp.layers.TransformerEncoder(num_heads=num_heads, dropout=dropout,
                    activation='gelu', intermediate_dim=dff, normalize_first=norm_first,
                    name=f'base_encoder_block_{i}')
        for i in range(num_enc_layers)]

    def call(self, input, training=False):
        mask = tf.reduce_any(tf.not_equal(input, 0), axis=2)

        asv_pos = self.asv_pos_emb(input)
        asv_pos = self.asv_norm(asv_pos)
        output = self.add([input, asv_pos])
        output = self.asv_dropout(output, training=training)
        for i in range(num_enc_layers):
            output = self.encoder_blocks[i](output, padding_mask=mask, training=training)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
                "dropout": self.dropout
        })
        return config

@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="Memory")
class Memory(tf.keras.layers.Layer):
    def __init__(self, num_heads, mem_rows, mem_vec_size, **kwargs):
        """
        M -> nxm (mem_rows, mem_vec)
        K -> bxhxm (batch, head, mem_vec)
        E -> bxhxm (batch, head, mem_vec)
        w -> bxhxn
        """
        super().__init__(name="memory_head", **kwargs)
        self.mem_vec_size = mem_vec_size
        self.num_heads = num_heads
        self.mem_rows = mem_rows
        self.flatten = tf.keras.layers.Flatten()
        self.k_dropout = tf.keras.layers.Dropout(0.05)
        self.e_dropout = tf.keras.layers.Dropout(0.05)
        self.a_dropout = tf.keras.layers.Dropout(0.05)
        self.k = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_heads*self.mem_vec_size, activation=tf.keras.activations.linear),
            tf.keras.layers.Reshape((self.num_heads, self.mem_vec_size))
        ])
        # self.r = tf.keras.Sequential([
        #     tf.keras.layers.Dense(self.num_heads*self.mem_vec_size, kernel_initializer="ones", activation=tf.keras.activations.linear),
        #     tf.keras.layers.Reshape((self.num_heads, self.mem_vec_size))
        # ])
        self.e = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_heads*self.mem_vec_size, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Reshape((self.num_heads, self.mem_vec_size))
        ])
        self.a = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_heads*self.mem_vec_size, activation=tf.keras.activations.sigmoid),
            tf.keras.layers.Reshape((self.num_heads, self.mem_vec_size))
        ])
        
    
    def _compute_row_norm(self, x):
        return tf.norm(x, axis=-1, keepdims=True)
    
    @tf.function
    def _compute_w(self, inputs, memory, training):
        """
        input -> x_t 
        k -> bxhxm
        M -> nxm (mem_rows, mem_vec
        """
        k = self.k(inputs)
        k_norm = k / self._compute_row_norm(k)
        k_norm = self.k_dropout(k_norm, training=training)

        M_norm = memory / self._compute_row_norm(memory)
        w = tf.einsum("...ij,...kj->...ik", k_norm, M_norm)
        w = tf.nn.softmax(w, axis=-1)
        return w
    
    def _erase_block(self, inputs, w, memory, training):
        e = self.e(inputs)
        e = self.e_dropout(e, training=training)
        erase = tf.einsum('bhn,bhmi->bnm', w, tf.expand_dims(e, axis=-1))
        erase_block = tf.math.multiply(memory, erase)
        return erase_block
    
    def _add_block(self, inputs, w, training):
        a = self.a(inputs)
        a = self.a_dropout(a, training=training)
        add = tf.einsum("bhn,bhmi->bnm", w, tf.expand_dims(a, axis=-1))
        return add
    
    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.float32, shape=[16, 128]),
                                  tf.TensorSpec(dtype=tf.float32, shape=[16, 128, 64]),
                                  tf.TensorSpec(dtype=tf.float32, shape=[16, 4, 128]),
                                  tf.TensorSpec(dtype=tf.bool, shape=())])
    def call(self, inputs, memory, w, training=False):
        old_w = self.flatten(w)
        inputs = tf.concat([old_w, inputs], axis=-1)
        w = self._compute_w(inputs, memory, training)
        erase_block = self._erase_block(inputs, w, memory, training)
        add_block = self._add_block(inputs, w, training)
        return memory - tf.multiply(memory, erase_block) + add_block, w

    def get_config(self):
        config = super().get_config()
        config.update({
            "mem_vec_size": self.mem_vec_size,
            "num_heads": self.num_heads,
            "mem_rows": self.mem_rows
        })
        return config
    
@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="ReadMemory")
class ReadMemory(tf.keras.layers.Layer):
    def __init__(self, num_heads, mem_rows, mem_vec_size, **kwargs):
        """
        M -> nxm (mem_rows, mem_vec)
        K -> bxhxm (batch, head, mem_vec)
        E -> bxhxm (batch, head, mem_vec)
        w -> bxhxn
        """
        super().__init__(name="read_memory_head", **kwargs)
        self.mem_vec_size = mem_vec_size
        self.num_heads = num_heads
        self.mem_rows = mem_rows
        self.flatten = tf.keras.layers.Flatten()
        self.r = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_heads*self.mem_vec_size, kernel_initializer="ones", activation=tf.keras.activations.linear),
            tf.keras.layers.Reshape((self.num_heads, self.mem_vec_size))
        ])
    
    @tf.function
    def _compute_row_norm(self, x):
        return tf.norm(x, axis=-1, keepdims=True)
    
    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.float32, shape=[16, 128]),
                                  tf.TensorSpec(dtype=tf.float32, shape=[16, 128, 64]),
                                  tf.TensorSpec(dtype=tf.float32, shape=[16, 4, 128]),
                                  tf.TensorSpec(dtype=tf.bool, shape=())])
    def call(self, inputs, memory, w, training=False):
        old_w = self.flatten(w)
        inputs = tf.concat([old_w, inputs], axis=-1)
        r = self.r(inputs)
        r_norm = r / self._compute_row_norm(r)
        M_norm = memory / self._compute_row_norm(memory)
        w = tf.einsum("...ij,...kj->...ik", r_norm, M_norm)
        w = tf.nn.softmax(w, axis=-1)

        read = tf.einsum("bhn,bnm->bhm", w, memory)
        return read, w

    def get_config(self):
        config = super().get_config()
        config.update({
            "mem_vec_size": self.mem_vec_size,
            "num_heads": self.num_heads,
            "mem_rows": self.mem_rows
        })
        return config

@tf.keras.saving.register_keras_serializable(package="amplicon_gpt", name="NeuralMemory")
class NeuralMemory(tf.keras.layers.Layer):
    def __init__(self, num_heads, mem_rows, mem_vec_size, **kwargs):
        super().__init__(name="asv_embedding", **kwargs)
        self.mem_vec_size = mem_vec_size
        self.num_heads = num_heads
        self.mem_rows = mem_rows
        self.controller = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)
        self.memory = Memory(num_heads, mem_rows, mem_vec_size)
        self.reader = ReadMemory(num_heads, mem_rows, mem_vec_size)

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.float32, shape=[16, None, 64]),
                                  tf.TensorSpec(dtype=tf.bool, shape=())])
    def call(self, input, training=False):
        time_steps = tf.shape(input)[1]+1
        output = self.controller(input, training=training)
        output = tf.transpose(output, [1,0,2])

        memory_output = tf.TensorArray(tf.float32, time_steps)
        memory_output = memory_output.write(0, tf.zeros((16, self.mem_rows, self.mem_vec_size), dtype=tf.float32))
        w_w = UnitUniform(self.mem_vec_size)((16, self.num_heads, self.mem_rows))

        read_output = tf.TensorArray(tf.float32, time_steps)
        read_output = read_output.write(0, tf.zeros((16, self.num_heads, self.mem_vec_size), dtype=tf.float32))
        w_r = UnitUniform(self.mem_vec_size)((16, self.num_heads, self.mem_rows))
        for i in tf.range(1, tf.shape(input)[1]):
            cur_mem = memory_output.read(i-1)            
            memory, w_w = self.memory(output[i-1], cur_mem, w_w, training=training)
            memory_output = memory_output.write(i, memory)

            read, w_r = self.reader(output[i-1], cur_mem, w_r, training=training)
            read_output = read_output.write(i, read)

        return tf.squeeze(tf.gather(read_output.stack(),[time_steps-1], axis=0))
        # return tf.squeeze(tf.gather(memory_output.stack(),[time_steps-1], axis=0))
        # return tf.squeeze(self.memory.read(output[-1], tf.squeeze(tf.gather(memory_output.stack(),[time_steps-1], axis=0)), w))

    def get_config(self):
        config = super().get_config()
        config.update({
            "mem_vec_size": self.mem_vec_size,
            "num_heads": self.num_heads,
            "mem_rows": self.mem_rows
        })
        return config
    