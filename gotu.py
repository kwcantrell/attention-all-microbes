import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.utils import data_utils
import tensorflow_models as tfm
from aam.callbacks import ProjectEncoder
from gotu.gotu_data_utils import generate_gotu_dataset, batch_dataset
from aam.layers import NucleotideEinsum
from aam.model_utils import _construct_base
from aam.callbacks import SaveModel

class CSVCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, gotu_map, out_dir, report_back_after_epochs=5):
        super().__init__()
        self.dataset = dataset
        self.gotu_map = gotu_map
        self.out_dir = out_dir
        self.report_back_after_epochs = report_back_after_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.report_back_after_epochs == 0:
            for x, y in self.dataset:
                encoder_inputs = x['encoder_inputs']
                decoder_inputs = x['decoder_inputs']
                true_gotus = y
                break
            
            pred_val = tf.math.argmax(self.model(x), axis=-1)
            # true_val = np.array([ys for (_,ys) in self.dataset])
            
            pred_df = pd.DataFrame(
                data=pred_val,
                columns=["predicted_gotus"]
            )
            pred_df.to_csv(os.path.join(self.out_dir, "we_tried.csv"))
            
            
class GOTU_Pad(tf.keras.layers.Layer):
    def __init__(self, gotu_size, **kwargs):
        super().__init__(**kwargs)
        self.gotu_size = gotu_size
        
    def call(self, inputs):
        if tf.is_symbolic_tensor(inputs):
            num_asvs = tf.shape(inputs)[1]
            inputs = tf.pad(inputs, [(0,0), (0, self.gotu_size - num_asvs), (0,0)])
        tf.print("After Padding", tf.shape(inputs))
        return inputs
            
    

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Configuration loading
with open("agp-data/base-configure.json") as f:
    config = json.load(f)

# Data preparation
asv_path = "../data/asv_ordered_table.biom"
gotu_path = "../data/gotu_ordered_table.biom"
dataset, gotu_dict = generate_gotu_dataset(gotu_path, asv_path)

size = dataset.cardinality().numpy()
batch_size = 8
train_size = int(size * 0.7 / batch_size) * batch_size

training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
training_dataset = batch_dataset(training_dataset, batch_size)
val_data = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
validation_dataset = batch_dataset(val_data, batch_size)

import tensorflow as tf

model = _construct_base(
    batch_size=8,
    dropout=0,
    pca_hidden_dim=64,
    pca_heads=4,
    pca_layers=1,
    dff=1024,
    d_model=128,
    enc_layers=6,
    enc_heads=8,
    output_dim=32,
    max_bp=150
)

VOCAB_SIZE = 5  
EMBEDDING_DIM = 128
SEQUENCE_LENGTH = 150
INTERMEDIATE_DIM = 1024
NUM_HEADS = 8
NUM_GOTUS = 6838

encoder_inputs = tf.keras.Input(shape=(None, 150), batch_size=8, dtype=tf.int64, name="encoder_inputs")
decoder_inputs = tf.keras.Input(shape=(None, 1), batch_size=8, dtype=tf.int64, name="decoder_inputs")
new_model = tf.keras.Model(
    inputs=model.inputs, 
    outputs=model.layers[-2].output
)
model_output = new_model(encoder_inputs)
# embedding_output = tf.keras.layers.Embedding(
#     input_dim=VOCAB_SIZE,  
#     output_dim=EMBEDDING_DIM,  
#     input_length=SEQUENCE_LENGTH, 
#     name="embedding"
# )(input_layer)

# normalized_input = tf.keras.layers.LayerNormalization()(embedding_output)

# model_output = NucleotideEinsum(128, input_max_length=150, normalize_output=True, reduce_tensor=True, activation='relu')(normalized_input)
decoder_embedding = tf.keras.layers.Embedding(
    NUM_GOTUS + 4,
    128,
    embeddings_initializer="uniform",
    input_length=1,
    name="decoder_embedding")(decoder_inputs)
decoder_embedding = tf.squeeze(decoder_embedding, axis=-2)

model_output = tfm.nlp.models.TransformerDecoder(
    num_layers=6,
    num_attention_heads=NUM_HEADS,
    intermediate_size=INTERMEDIATE_DIM,
    norm_first=True,
    activation='relu',
)(decoder_embedding, model_output)
# model_output = GOTU_Pad(NUM_GOTUS + 3)(model_output)
# model_output = tf.keras.layers.Dropout(0.5)(model_output)

decoder_outputs = tf.keras.layers.Dense(NUM_GOTUS + 4)(model_output)
gotu_model = tf.keras.Model(inputs=(encoder_inputs, decoder_inputs), outputs=decoder_outputs, name="asv_to_gotu_classifier")
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
gotu_model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])

gotu_model.summary()
# for x, y in training_dataset:
#     print(gotu_model(x))
#    # print(y)
#     break
history = gotu_model.fit(training_dataset, validation_data=validation_dataset, callbacks=[SaveModel("gotu_decoder_model"), CSVCallback(validation_dataset, gotu_dict, "gotu_decoder_model/")], epochs=10)


### Transfer Learning ###
# model = transfer_learn_base(batch_size=8, dropout=0.5)
# model.load_weights("base-model/encoder.keras")
# for layer in model.layers:
#     layer.trainable = False
# gotu_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

# # Model summary and verification
# model.summary()
# gotu_model.summary()


# # Check base model is non-trainable
# for layer in gotu_model.layers:
#     print(layer.name, layer.trainable, layer.output_shape)
# assert all(not layer.trainable for layer in gotu_model.layers), "Some layers are still trainable."

