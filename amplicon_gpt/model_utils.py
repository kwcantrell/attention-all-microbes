import os
import numpy as np
import pandas as pd
from biom import load_table
import tensorflow as tf
import keras_nlp
from tensorflow_models import nlp # need for PositionEmbedding without cannot load base_model
from amplicon_gpt.losses import unifrac_loss_var, _pairwise_distances # need for unifrac_loss_var without cannot load base_model
from amplicon_gpt.losses import regression_loss_variance, regression_loss_difference_in_means, regression_loss_combined, regression_loss_normal
from amplicon_gpt.layers import SampleEncoder,  NucleotideEinsum, ReadHead

MAX_SEQ = 1600
BATCH_SIZE=2

def transfer_learn_base(sequence_tokenizer, lstm_seq_out, batch_size, max_num_per_seq, dropout, root_path, load_prev_path=False, **kwargs):   
    d_model = 64
    dff = 512
    num_heads = 6
    num_enc_layers = 4
    norm_first = False
    
    input = tf.keras.Input(shape=[None,100], batch_size=batch_size, dtype=tf.int64)
    model_input = tf.keras.layers.Embedding(
        5,
        d_model,
        input_length=100,
        input_shape=[batch_size, None, 100],
        name="embedding")(input)
    model_input = NucleotideEinsum(d_model, use_pos_emb=True)(model_input)
    model_input = SampleEncoder(dropout, num_enc_layers, num_heads, dff, norm_first)(model_input)
    output = ReadHead(64)(model_input)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model
    
@tf.keras.saving.register_keras_serializable(package="amplicon_gpt.metrics")
class MAE(tf.keras.metrics.Metric):
    def __init__(self, name='mae_loss', dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        self.loss = self.add_weight(name='rl', initializer='zero', dtype=tf.float32)
        self.i = self.add_weight(name='i', initializer='zero', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.loss.assign_add(tf.reduce_sum(tf.abs(_pairwise_distances(y_pred)-y_true)))
        self.i.assign_add(tf.constant(((BATCH_SIZE*BATCH_SIZE)-BATCH_SIZE) / 2.0, dtype=tf.float32))

    def result(self):
        return self.loss / self.i
    
def compile_model(model):
    initial_learning_rate = 1.5e-3
    decay_steps = 10400.0
    decay_rate = 0.7
    lr = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate, decay_steps, decay_rate)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, beta_2=0.98, epsilon=1e-7)
    model.compile(
        optimizer=optimizer,
        loss=unifrac_loss_var, metrics=[MAE()],
        jit_compile=False)
    return model


def load_full_base_model(base_model_path, **kwargs):
    base_model = tf.keras.models.load_model(base_model_path)
    base_model.trainable = False
    return base_model

def get_base_model(base_model_path, **kwargs):
    """
    Returns the base model input layer and the output of the 'community level' encoder (i.e. the transformer encoder block).
    This will also disable all trainable parameters.
    """
    base_model = tf.keras.models.load_model(base_model_path)
    base_model.trainable = False
    input = base_model.inputs[0] # base model only has one input however, .inputs returns list
    base_output = base_model.get_layer('base_encoder_block_3').output
    return input, base_output
