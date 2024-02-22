import click
import json
import os
import tensorflow as tf
import amplicon_gpt._parameter_descriptions as desc
from amplicon_gpt.callbacks import ProjectEncoder
from amplicon_gpt.data_utils import (
    get_sequencing_dataset, get_unifrac_dataset, combine_seq_dist_dataset, batch_dist_dataset, batch_label_dataset, combine_label_dataset, _get_filtered_table_and_metadata
)
from amplicon_gpt.model_utils import transfer_learn_base, compile_model, classification, regression
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from biom import load_table

# Allow using -h to show help information
# https://click.palletsprojects.com/en/7.x/documentation/#help-parameter-customization
CTXSETS = {"help_option_names": ["-h", "--help"]}

@click.group()
def transfer_learning():
    pass

@transfer_learning.command('unifrac')
@click.option(
    '--config-json',
    required=True,
    type=click.Path(exists=True),
    help=desc.CONFIG_JSON
)
@click.option(
    '-c', '--continue-training',
    required=False, default=False, is_flag=True,
    help=desc.CONTINUE_TRAINING, 
)
@click.option(
    '--output-model-summary',
    required=False, default=False, is_flag=True,
    help=desc.OUTPUT_MODEL_SUMMARY
)
def unifrac(config_json, continue_training, output_model_summary):
    with open(config_json) as f:
        config = json.load(f)

    seq_dataset = get_sequencing_dataset(**config)
    unifrac_dataset = get_unifrac_dataset(**config)
    sequence_tokenizer = tf.keras.layers.TextVectorization(max_tokens=7, split='character', output_mode='int', output_sequence_length=100)
    sequence_tokenizer.adapt(seq_dataset.take(1))
    dataset, proj_dataset = combine_seq_dist_dataset(seq_dataset, unifrac_dataset, **config)

    size = seq_dataset.cardinality().numpy()
    batch_size = config['batch_size']
    train_size = int(size*config['train_percent']/batch_size)*batch_size

    training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
    training_dataset = batch_dist_dataset(training_dataset, shuffle=True, **config)
    
    val_data = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = batch_dist_dataset(val_data, **config)

    model = transfer_learn_base(batch_size=batch_size, dropout=config['dropout'])
    # model = classification(batch_size=batch_size, dropout=config['dropout'])
    model = compile_model(model)
    for x, _ in training_dataset.take(1):
        y = model(x)
    
    if output_model_summary:
        model.summary()
    
    if 'patience' in config:
        patience=config['patience']
    else:
        patience=10
    config['repeat'] = 1
    
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                           patience=5, min_lr=0.001)
    # Define the Keras TensorBoard callback.
    logdir="base-model/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)

    model.fit(
        training_dataset, validation_data=validation_dataset,
        epochs=config['epochs'], initial_epoch=0, batch_size=config['batch_size'],
        callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', start_from_epoch=0, patience=patience, mode='min'),
                    ProjectEncoder(proj_dataset.padded_batch(config['batch_size']),**config),
                    tensorboard_callback
        ]
    )
    # model.save(os.path.join(config['root_path'], 'model.keras'), save_format='keras')

@transfer_learning.command('classify')
@click.option(
    '--config-json',
    required=True,
    type=click.Path(exists=True),
    help=desc.CONFIG_JSON
)
def classify(config_json):
    with open(config_json) as f:
        config = json.load(f)

    table = load_table('agp-data/pruned_feature_table.biom')
    table = table.subsample(5000)
    metadata = pd.read_csv('agp-data/metadata.csv',sep=',', index_col='sample_id')
    filtered_table, filtered_metadata = table.align_to_dataframe(metadata, axis='sample')
    seq_dataset = get_sequencing_dataset(filtered_table)
    labels = filtered_metadata.month.to_numpy().astype('int') - 1
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = combine_label_dataset(seq_dataset, label_dataset)

    size = seq_dataset.cardinality().numpy()
    batch_size = 8
    train_size = int(size*0.8/batch_size)*batch_size

    training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
    training_dataset = batch_label_dataset(training_dataset, batch_size=batch_size, shuffle=True)
    
    val_data = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = batch_label_dataset(val_data, batch_size=batch_size)
    
    model = classification(batch_size, 0)

#     model = transfer_learn_base(batch_size=batch_size, dropout=config['dropout'])
#     model = compile_model(model)
#     for x, _ in training_dataset.take(1):
#         y = model(x)
    
#     model = compile_model(model)
#     for x, _ in training_dataset.take(1):
#         y = model(x)
    
    model.summary()
    
    if 'patience' in config:
        patience=config['patience']
    else:
        patience=10
    config['repeat'] = 1
    
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                           patience=5, min_lr=0.001)
    # Define the Keras TensorBoard callback.
    logdir="base-model/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)

    model.fit(
        training_dataset, validation_data=validation_dataset,
        epochs=config['epochs'], initial_epoch=0, batch_size=config['batch_size'],
        callbacks=[
            tensorboard_callback
        ]
    )
    # model.save(os.path.join(config['root_path'], 'model.keras'), save_format='keras')

@transfer_learning.command('regress')
@click.option(
    '--table-path',
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    '--metadata-path',
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    '--variable-to-regress',
    required=True,
)
@click.option(
    '--repeats-per-epoch',
    required=False, default=1
)
@click.option(
    '--batch-size',
    required=False, default=8
)
@click.option(
    '--train-percent',
    required=False, default=0.8
)
@click.option(
    '--epochs',
    required=False, default=100
)
@click.option(
    '--output-model-summary',
    required=False, default=False, is_flag=True,
    help=desc.OUTPUT_MODEL_SUMMARY
)
def regress(
        table_path,
        metadata_path,
        variable_to_regress,
        repeats_per_epoch,
        batch_size,
        train_percent,
        epochs,
        output_model_summary
    ):
    filtered_table, filtered_metadata = _get_filtered_table_and_metadata(
        table_path, metadata_path, variable_to_regress
    )
    
    seq_dataset = get_sequencing_dataset(filtered_table)
    unifrac_dataset = filtered_metadata.host_age.to_numpy()
    unifrac_dataset = tf.data.Dataset.from_tensor_slices(unifrac_dataset)
    dataset = combine_label_dataset(seq_dataset, unifrac_dataset)

    size = seq_dataset.cardinality().numpy()
    train_size = int(size*train_percent/batch_size)*batch_size

    training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
    training_dataset = batch_label_dataset(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        repeat=repeats_per_epoch
    )
    
    val_data = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = batch_label_dataset(val_data, batch_size=batch_size)

    model = regression(batch_size=batch_size)
    
    if output_model_summary:
        model.summary()
    
    patience=10
    
    # Define the Keras TensorBoard callback.
    logdir="base-model/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)

    model.fit(
        training_dataset, validation_data=validation_dataset,
        epochs=epochs, initial_epoch=0, batch_size=batch_size,
        callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', start_from_epoch=0, patience=patience, mode='min'),
                    tensorboard_callback
        ]
    )
    # model.save(os.path.join(config['root_path'], 'model.keras'), save_format='keras')

    
def main():
    transfer_learning(prog_name='transfer_learning')

if __name__ == '__main__':
    main()