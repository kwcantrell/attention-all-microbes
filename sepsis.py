import pandas as pd
import numpy as np
from aam.callbacks import SaveModel
from aam.layers import PCAProjector, ReadHead
from aam.metrics import MAE
from sepsis.model import AttentionRegression, _construct_model
from sepsis.layers import FeatureEmbedding, PCA, ProjectDown, BinaryLoadings
from sepsis.callbacks import MAE_Scatter, AvgFeatureConfidence
from sepsis.data_utils import (
    load_biom_table, create_rclr_table, convert_table_to_dataset,
    convert_to_normalized_dataset, filter_and_reorder, extract_col,
    batch_dataset, shuffle_table, train_val_split
)
import tensorflow as tf
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir= /opt/tensorflow_2.14_3.9/lib/python3.9/site-packages/nvidia/"

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

table_fp = '/home/kcantrel/amplicon-gpt/agp-data/stool-table.biom'
rclr_fp = '/home/kcantrel/amplicon-gpt/agp-data/stool-table.biom'
# create_rclr_table(table_fp, rclr_fp)

batch_size = 32
repeat = 5

orig_table = load_biom_table(table_fp)
rclr_table = shuffle_table(load_biom_table(rclr_fp))
feature_dataset = convert_table_to_dataset(rclr_table)
metadata = pd.read_csv(
    '/home/kcantrel/amplicon-gpt/agp-data/stool-meta.tsv', sep='\t',
    index_col=0
)
# ids = ['.'.join(id.split('.')[1:]) for id in rclr_table.ids(axis='sample')]
ids = rclr_table.ids(axis='sample')
metadata = filter_and_reorder(metadata, ids)
age_in_weeks = extract_col(metadata, 'host_age', output_dtype=np.float32)
age_dataset, mean_age, std_age = convert_to_normalized_dataset(age_in_weeks)
full_dataset = tf.data.Dataset.zip((feature_dataset, age_dataset))
training, validation = train_val_split(
    full_dataset,
    train_percent=1.
)
training_size = training.cardinality().numpy()
training_ids = ids[:training_size]
validation_ids = ids[training_size:]

training_dataset = batch_dataset(
    training,
    batch_size,
    repeat=repeat,
    shuffle=True
)
validation_dataset = batch_dataset(
    validation,
    batch_size,
    shuffle=False
)
training_no_shuffle = batch_dataset(
    training,
    batch_size,
    shuffle=False
)

token_dim = 512
features_to_add = .1
dropout = .10
d_model = 1024
ff_dim = 32
report_back_after = 10
epochs = 1000
model = _construct_model(
    rclr_table.ids(axis='observation').tolist(),
    mean_age,
    std_age,
    token_dim,
    features_to_add,
    dropout,
    d_model,
    ff_dim,
)

reg_out_callbacks = [
    MAE_Scatter(
        'training',
        training_no_shuffle,
        metadata[metadata.index.isin(training_ids)],
        'host_age',
        None,
        None,
        mean_age,
        std_age,
        'base-model-age/figures',
        report_back_after=report_back_after
    )
]

emb_out_callbacks = [
    # AvgFeatureConfidence(
    #     'training',
    #     training_no_shuffle,
    #     metadata[metadata.index.isin(training_ids)],
    #     'host_age',
    #     'Age',
    #     'intervention_group',
    #     'Intervention',
    #     'sepsis/figures',
    #     report_back_after=report_back_after
    # ),
]

core_callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        "loss",
        factor=0.5,
        patients=2,
        min_lr=0.000001
    ),
    tf.keras.callbacks.EarlyStopping(
        'loss',
        patience=50
    ),
    SaveModel("base-model-age")
]
model.fit(
    training_dataset,
    callbacks=[
        *reg_out_callbacks,
        *core_callbacks
    ],
    epochs=epochs
)
