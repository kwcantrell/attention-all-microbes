import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow_models as tfm
from aam.layers import ReadHead
from aam.losses import denormalize, pairwise_residual_mse, mae_loss, pairwise_residual_mae
from aam.metrics import MAE
from sepsis.callbacks import mae_scatter
from sepsis.layers import FeatureEmbedding
from sepsis.data_utils import (
    load_biom_table, create_rclr_table, convert_table_to_dataset,
    convert_to_normalized_dataset, filter_and_reorder, extract_col,
    shuffle_table, train_val_split, batch_dataset
)

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

emb_dim = 128
enc_layers = 4
enc_heads = 8
dff = 1024
dropout = 0.5

lr = 0.0001
epochs = 1000
batch_size = 8

table_fp = 'sepsis/data/pangenome_zebra_prev_LP2_trial_ft.biom'
rclr_fp = 'sepsis/data/pangenome_zebra_prev_LP2_trial_ft_rclr.biom'
create_rclr_table(table_fp, rclr_fp)
orig_table = load_biom_table(table_fp)
rclr_table = shuffle_table(load_biom_table(rclr_fp))
feature_dataset = convert_table_to_dataset(rclr_table)
metadata = pd.read_csv('sepsis/data/15004_20230703-113400.txt', sep='\t',
                       index_col=0)
ids = ['.'.join(id.split('.')[1:]) for id in rclr_table.ids(axis='sample')]
metadata = filter_and_reorder(metadata, ids)
age_in_weeks = extract_col(metadata, 'host_age', output_dtype=np.float32)
age_dataset, mean_age, std_age = convert_to_normalized_dataset(age_in_weeks)
full_dataset = tf.data.Dataset.zip((feature_dataset, age_dataset))
training, validation = train_val_split(full_dataset,
                                       train_percent=.8)
training_size = training.cardinality().numpy()
training_ids = ids[:training_size]
validation_ids = ids[training_size:]

training_dataset = batch_dataset(training, batch_size, shuffle=True)
validation_dataset = batch_dataset(validation, batch_size, shuffle=False)
training_no_shuffle = batch_dataset(training, batch_size, shuffle=False)
feature_input = tf.keras.Input(shape=[None],
                               dtype=tf.string,
                               name="feature")
rclr_input = tf.keras.Input(shape=[None, 1],
                            dtype=tf.float32,
                            name="rclr")

output = FeatureEmbedding(
    rclr_table.ids(axis='observation'),
    emb_dim)((feature_input, rclr_input))
output = tf.keras.layers.Dropout(dropout)(output)
output = tfm.nlp.models.TransformerEncoder(
    num_layers=enc_layers,
    num_attention_heads=enc_heads,
    intermediate_size=dff,
    dropout_rate=dropout,
    norm_first=True,
    activation='relu',
)(output)
output = ReadHead(
    hidden_dim=emb_dim,
    num_heads=8,
    num_layers=1,
    output_dim=1,
    dropout=dropout
)(output)
model = tf.keras.Model(inputs=(feature_input, rclr_input), outputs=output)

optimizer = tf.keras.optimizers.AdamW(learning_rate=lr,
                                      beta_1=0.9,
                                      beta_2=0.98,
                                      epsilon=1e-9,
                                      weight_decay=0.001)
model.compile(optimizer=optimizer,
            #   loss=pairwise_residual_mae(batch_size, mean_age, std_age),
              loss='mae',
              metrics=[MAE(mean_age, std_age, dtype=tf.float32)],
              jit_compile=False)
model.summary()
model.fit(training_dataset,
          validation_data=validation_dataset,
          callbacks=[mae_scatter(mean_age,
                                 std_age,
                                 'validation',
                                 validation_dataset,
                                 metadata,
                                 'sepsis/data-mae'),
                     mae_scatter(mean_age,
                                 std_age,
                                 'training',
                                 training_no_shuffle,
                                 metadata,
                                 'sepsis/data-mae')],
          epochs=epochs,
          batch_size=batch_size)
for i, (inputs, age) in enumerate(training_dataset):
    if i > 5:
        break
    x1 = model(inputs)
    print(denormalize(x1, mean_age, std_age))
    print(denormalize(age, mean_age, std_age))
    break