import tensorflow as tf
import pandas as pd
import numpy as np
from aam.callbacks import SaveModel
from aam.metrics import MAE
from sepsis.model import AttentionRegression, _construct_model
from sepsis.callbacks import MAE_Scatter, AvgFeatureConfidence
from sepsis.data_utils import (
    load_biom_table, create_rclr_table, convert_table_to_dataset,
    convert_to_normalized_dataset, filter_and_reorder, extract_col,
    batch_dataset, shuffle_table, train_val_split
)
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

table_fp = 'sepsis/data/pangenome_zebra_prev_LP2_trial_ft.biom'
rclr_fp = 'sepsis/data/pangenome_zebra_prev_LP2_trial_ft_rclr-2.biom'
create_rclr_table(table_fp, rclr_fp)

batch_size = 32
repeat = 5

orig_table = load_biom_table(table_fp)
rclr_table = shuffle_table(load_biom_table(rclr_fp))
feature_dataset = convert_table_to_dataset(rclr_table)
metadata = pd.read_csv(
    'sepsis/data/full-intervention-withgroup.tsv', sep='\t',
    index_col=0
)
ids = ['.'.join(id.split('.')[1:]) for id in rclr_table.ids(axis='sample')]
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
features_to_add = .5
dropout = .10
d_model = 1024
ff_dim = 32
report_back_after = 10
epochs = 1000
ids = rclr_table.ids(axis='observation').tolist()

model = _construct_model(
    ids,
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
        'intervention_group',
        'Intervention',
        mean_age,
        std_age,
        'sepsis/figures',
        report_back_after=report_back_after
    )
]

core_callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        "loss",
        factor=0.5,
        patients=2,
        min_lr=0.00001
    ),
    tf.keras.callbacks.EarlyStopping(
        'confidence',
        patience=50
    ),
    SaveModel("sepsis/model-3")
]
model.fit(
    training_dataset,
    callbacks=[
        {"reg_out": reg_out_callbacks},
        *core_callbacks
    ],
    epochs=epochs
)
