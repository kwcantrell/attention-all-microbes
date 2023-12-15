import pandas as pd
import numpy as np
import tensorflow as tf
from simple_transformer import TransformerBlock, TokenEmbedding
from biom import load_table
from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from keras.models import Sequential, Model

def get_global_constant(constant):
    constants = {
        "BASE_16S_TABLE":'dataset/training/data/16s-full-train.biom',
    }
    return constants[constant]

def get_observation_vocab(table_path):
    table = get_table(table_path)
    return table.ids(axis='observation')

def get_table(table_path):
    return load_table(table_path)