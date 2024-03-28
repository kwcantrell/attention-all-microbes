import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow_models as tfm
from aam.layers import ReadHead
from aam.losses import pairwise_residual_mse
from sepsis.callbacks import mae_scatter
from sepsis.layers import FeatureEmbedding
from sepsis.metrics import mae, denormalize
from sepsis.data_utils import (
    load_biom_table, create_rclr_table, convert_table_to_dataset,
    convert_to_normalized_dataset, filter_and_reorder, extract_col,
    batch_dataset, shuffle_table, train_val_split
)

table_fp = 'sepsis/data/pangenome_zebra_prev_LP2_trial_ft.biom'
rclr_fp = 'sepsis/data/pangenome_zebra_prev_LP2_trial_ft_rclr.biom'
create_rclr_table(table_fp, rclr_fp)
orig_table = load_biom_table(table_fp)
rclr_table = shuffle_table(load_biom_table(rclr_fp))

feature_dataset = convert_table_to_dataset(rclr_table)

metadata = pd.read_csv('sepsis/data/15004_20230703-113400.txt', sep='\t', index_col=0)
ids = ['.'.join(id.split('.')[1:]) for id in rclr_table.ids(axis='sample')]
metadata = filter_and_reorder(metadata, ids)
age_in_weeks = extract_col(metadata, 'host_age', output_dtype=np.float32)
age_dataset, mean_age, std_age = convert_to_normalized_dataset(age_in_weeks)

full_dataset = tf.data.Dataset.zip((feature_dataset, age_dataset))
training_dataset, validation_dataset = train_val_split(full_dataset, train_percent=.8, batch_size=8)

emb_vocab = ids
emb_dim = 128
enc_layers = 4
enc_heads = 8
dff = 1024
dropout = 0.50

lr = 0.0001
epochs = 1000
batch_size = 8

feature_input = tf.keras.Input(shape=[None], batch_size=batch_size, dtype=tf.string, name="feature")
rclr_input = tf.keras.Input(shape=[None, 1], batch_size=batch_size, dtype=tf.float32, name="rclr")

output = FeatureEmbedding(rclr_table.ids(axis='observation'), emb_dim)((feature_input, rclr_input))
output = tfm.nlp.models.TransformerEncoder(
    num_layers=enc_layers,
    num_attention_heads=enc_heads,
    intermediate_size=dff,
    attention_dropout_rate=0.50,
    intermediate_dropout=0.5,
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

optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                          beta_1=0.9,
                                          beta_2=0.98,
                                          epsilon=1e-9)
model.compile(optimizer=optimizer,
                  loss=pairwise_residual_mse(batch_size),
                  metrics=[mae(batch_size, mean_age, std_age)],
                  jit_compile=False)
model.fit(training_dataset,
          validation_data=validation_dataset,
          callbacks=[mae_scatter(mean_age, std_age, 'validation', validation_dataset, 'sepsis/data'),
                     mae_scatter(mean_age, std_age, 'training', training_dataset, 'sepsis/data')],
          epochs=epochs,
          batch_size=batch_size)
for i, (inputs, age) in enumerate(training_dataset):
    if i > 5:
        break
    x1 = model(inputs)
    print(denormalize(x1, mean_age, std_age))
    print(denormalize(age, mean_age, std_age))
    break