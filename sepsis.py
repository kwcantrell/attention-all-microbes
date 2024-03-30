import tensorflow as tf
import pandas as pd
import numpy as np
from biom import load_table
from biom.util import biom_open
# from aam.losses import MeanSquaredError, PairwiseMSE
from aam.losses import MeanSquaredError, pairwise_residual_mse
from aam.metrics import MAE
from sepsis.layers import FeatureEmbedding, PCA, ProjectDown
from sepsis.callbacks import MAE_Scatter, ViolinPrediction, ViolinResiduals
from sepsis.data_utils import (
    load_biom_table, create_rclr_table, convert_table_to_dataset,
    convert_to_normalized_dataset, filter_and_reorder, extract_col,
    batch_dataset, shuffle_table, train_val_split
)
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

table_fp = 'sepsis/data/pangenome_zebra_prev_LP2_trial_ft.biom'
rclr_fp = 'sepsis/data/pangenome_zebra_prev_LP2_trial_ft_rclr.biom'
# create_rclr_table(table_fp, rclr_fp)

emb_dim = 16
enc_layers = 4
enc_heads = 8
dff = 1024
dropout = 0.3

lr = 0.00001
epochs = 1000
batch_size = 32

orig_table = load_biom_table(table_fp)
rclr_table = shuffle_table(load_biom_table(rclr_fp))
feature_dataset = convert_table_to_dataset(rclr_table)
metadata = pd.read_csv('sepsis/data/full-intervention-withgroup.tsv', sep='\t',
                       index_col=0)
ids = ['.'.join(id.split('.')[1:]) for id in rclr_table.ids(axis='sample')]
metadata = filter_and_reorder(metadata, ids)
age_in_weeks = extract_col(metadata, 'host_age', output_dtype=np.float32)
age_dataset, mean_age, std_age = convert_to_normalized_dataset(age_in_weeks)
full_dataset = tf.data.Dataset.zip((feature_dataset, age_dataset))
training, validation = train_val_split(full_dataset,
                                       train_percent=.7)
training_size = training.cardinality().numpy()
training_ids = ids[:training_size]
validation_ids = ids[training_size:]

training_dataset = batch_dataset(training, batch_size, shuffle=True)
validation_dataset = batch_dataset(validation, batch_size, shuffle=False)
training_no_shuffle = batch_dataset(training, batch_size, shuffle=False)

token_dim = 512
dropout = .30
emb_dim = 64
report_back_after = 30

feature_input = tf.keras.Input(shape=[None],
                               dtype=tf.string,
                               name="feature")
rclr_input = tf.keras.Input(shape=[None],
                            dtype=tf.float32,
                            name="rclr")

output = FeatureEmbedding(
    token_dim,
    rclr_table.ids(axis='observation'),
    emb_dim,
    dropout)((feature_input, rclr_input))
output = PCA(emb_dim)(output)
output = ProjectDown(emb_dim,
                     dims=3,
                     reduce_dim=True)(output)
output = ProjectDown(emb_dim,
                     dims=2,
                     reduce_dim=False)(output)
model = tf.keras.Model(inputs=(feature_input, rclr_input), outputs=output)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01,
                                      beta_1=0.9,
                                      beta_2=0.98,
                                      epsilon=1e-9)
model.compile(optimizer=optimizer,
            #   loss=MeanSquaredError(mean_age, std_age),
              # loss=pairwise_residual_mse(batch_size, mean_age, std_age),
              # loss=pairwise_residual_mse(batch_size),
              loss="mse",
              metrics=[MAE(mean_age, std_age, dtype=tf.float32)],
              jit_compile=False)
model.summary()
model.fit(training_dataset,
          validation_data=validation_dataset,
          callbacks=[MAE_Scatter('validation',
                                 validation_dataset,
                                 metadata[metadata.index.isin(validation_ids)],
                                 'host_age',
                                 'intervention_group',
                                 'Intervention',
                                 mean_age,
                                 std_age,
                                 'sepsis/figures',
                                 report_back_after=report_back_after),
                     MAE_Scatter('training',
                                 training_no_shuffle,
                                 metadata[metadata.index.isin(training_ids)],
                                 'host_age',
                                 'intervention_group',
                                 'Intervention',
                                 mean_age,
                                 std_age,
                                 'sepsis/figures',
                                 report_back_after=report_back_after),
                     ViolinResiduals('validation',
                                 validation_dataset,
                                 metadata[metadata.index.isin(validation_ids)],
                                 'host_age',
                                 'Age',
                                 'intervention_group',
                                 'Intervention',
                                 mean_age,
                                 std_age,
                                 'sepsis/figures',
                                 report_back_after=report_back_after),
                     ViolinResiduals('training',
                                 training_no_shuffle,
                                 metadata[metadata.index.isin(training_ids)],
                                 'host_age',
                                 'Age',
                                 'intervention_group',
                                 'Intervention',
                                 mean_age,
                                 std_age,
                                 'sepsis/figures',
                                 report_back_after=report_back_after),
                     # tf.keras.callbacks.EarlyStopping('val_mae',
                     #                                  patience=200,
                     #                                  restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau("val_mae", factor=0.1, patients=0, cooldown=5)],
          epochs=epochs)

validation_meta = metadata[metadata.index.isin(validation_ids)]
validation_age = validation_meta['host_age'].to_list()
validation_inter = validation_meta['intervention_group'].to_list()
violinplot_residuals(
                validation_dataset,
                validation_age,
                'Age',
                validation_inter,
                'Intervention',
                model,
                fname=os.path.join(
                    'sepsis/figures', f'Final-Violin-validation.png'
                    ),
                mean=mean_age,
                std=std_age
            )