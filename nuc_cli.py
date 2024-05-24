import click
import tensorflow as tf
import aam._parameter_descriptions as desc
from aam.cli_util import aam_model_options
from aam.data_utils import (
    batch_dataset,
    get_unifrac_dataset, combine_datasets, get_sequencing_dataset,
    get_sequencing_count_dataset, combine_count_datasets
)
from attention_regression.data_utils import (
    load_biom_table, shuffle_table, train_val_split
)
from aam.model_utils_current import pretrain_unifrac, regressor
from attention_regression.callbacks import MAE_Scatter
from aam.callbacks import SaveModel
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import biom
from typing import Tuple
from attention_regression.data_utils import filter_and_reorder, extract_col, convert_to_normalized_dataset
from bp import parse_newick
from aam.nuc_model import TransferLearnNucleotideModel


@click.group()
class cli:
    pass


def _create_dataset(
    i_table_path: str,
    i_tree_path: str,
    i_max_bp: str,
    p_batch_size: int,
):
    table = shuffle_table(load_biom_table(i_table_path))

    # check for mismatch samples
    ids = table.ids(axis='sample')

    # TODO: check for invalid metadata values
    table = table.remove_empty(axis='observation', inplace=False)
    feature_dataset = get_sequencing_dataset(table)
    distance_dataset = get_unifrac_dataset(i_table_path, i_tree_path)

    full_dataset = combine_datasets(
        feature_dataset,
        distance_dataset,
        i_max_bp,
        add_index=True
    )
    training, val = train_val_split(
        full_dataset,
        train_percent=.8
    )

    return {
        'training': training,
        'val': val,
        'sample_ids': ids,
        'table': table,
    }


def _aam_globals():
    return {
        'feature-attention-methods': ['add_features', 'mask_features', 'none']
    }


aam_globals = _aam_globals()


@cli.command()
@click.option(
    '--i-table-path',
    required=True,
    help=desc.TABLE_DESC,
    type=click.Path(exists=True)
)
@click.option(
    '--i-metadata-path',
    required=True,
    help=desc.TABLE_DESC,
    type=click.Path(exists=True)
)
@click.option(
    '--i-metadata-col',
    required=True,
    type=str
)
@click.option(
    '--i-max-bp',
    required=True,
    type=int
)
@click.option(
    '--p-missing-samples',
    default='error',
    type=click.Choice(['error', 'ignore'], case_sensitive=False),
    help=desc.MISSING_SAMPLES_DESC
)
@click.option(
    '--p-batch-size',
    default=8,
    show_default=True,
    type=int
)
@click.option(
    '--p-epochs',
    default=1000,
    show_default=True,
    type=int
)
@click.option(
    '--p-repeat',
    default=5,
    show_default=True,
    type=int
)
@click.option(
    '--p-dropout',
    default=0.01,
    show_default=True,
    type=float
)
@click.option(
    '--p-token-dim',
    default=512,
    show_default=True,
    type=int
)
@click.option(
    '--p-feature-attention-method',
    default='add_features',
    type=click.Choice(aam_globals['feature-attention-methods'])
)
@click.option(
    '--p-features-to-add-rate',
    default=1.,
    show_default=True,
    type=float
)
@click.option(
    '--p-ff-d-model',
    default=128,
    show_default=True,
    type=int
)
@click.option(
    '--p-ff-clr',
    default=64,
    show_default=True,
    type=int
)
@click.option(
    '--p-pca-heads',
    default=8,
    show_default=True,
    type=int
)
@click.option(
    '--p-enc-layers',
    default=2,
    show_default=True,
    type=int
)
@click.option(
    '--p-enc-heads',
    default=8,
    show_default=True,
    type=int
)
@click.option(
    '--p-lr',
    default=0.01,
    show_default=True,
    type=float
)
@click.option(
    '--p-report-back-after',
    default=5,
    show_default=True,
    type=int
)
@click.option(
    '--p-output-dir',
    required=True
)
def fit_regressor(
    i_table_path: str,
    i_metadata_path: str,
    i_metadata_col: str,
    i_max_bp: int,
    p_missing_samples: str,
    p_batch_size: int,
    p_epochs: int,
    p_repeat: int,
    p_dropout: float,
    p_token_dim: int,
    p_feature_attention_method: str,
    p_features_to_add_rate: float,
    p_ff_d_model: int,
    p_ff_clr: int,
    p_pca_heads: int,
    p_enc_layers: int,
    p_enc_heads: int,
    p_lr: float,
    p_report_back_after: int,
    p_output_dir: str
):
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)

    figure_path = os.path.join(p_output_dir, 'figures')
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    table = shuffle_table(load_biom_table(i_table_path))
    metadata = pd.read_csv(
        i_metadata_path, sep='\t',
        index_col=0
    )

    # check for mismatch samples
    ids = table.ids(axis='sample')
    shared_ids = set(ids).intersection(set(metadata.index))
    min_ids = min(len(shared_ids), len(ids), len(metadata.index))
    max_ids = max(len(shared_ids), len(ids), len(metadata.index))
    if len(shared_ids) == 0:
        raise Exception('Table and Metadata have no matching sample ids')
    if min_ids != max_ids and p_missing_samples == 'error':
        raise Exception('Table and Metadata do share all same sample ids.')
    elif min_ids != max_ids and p_missing_samples == 'ignore':
        print('Warning: Table and Metadata do share all same sample ids.')
        print('Table and metadata will be filtered')
        table = table.filter(shared_ids, inplace=False)
        metadata = metadata[metadata.index.isin(shared_ids)]
        ids = table.ids(axis='sample')

    # TODO: check for invalid metadata values
    table = table.remove_empty(axis='observation', inplace=False)
    feature_dataset = get_sequencing_dataset(table)
    feature_count_dataset = get_sequencing_count_dataset(table)
    metadata = filter_and_reorder(metadata, ids)
    regression_data = extract_col(
        metadata,
        i_metadata_col,
        output_dtype=np.float32
    )
    regression_dataset, mean, std = convert_to_normalized_dataset(
        regression_data,
        'z'
    )
    full_dataset = combine_count_datasets(
        feature_dataset,
        feature_count_dataset,
        regression_dataset,
        i_max_bp,
        add_index=True
    )
    training, val = train_val_split(
        full_dataset,
        train_percent=.8
    )
    training_size = training.cardinality().numpy()
    training_ids = ids[:training_size]
    training_dataset = batch_dataset(
        training,
        p_batch_size,
        i_max_bp,
        shuffle=True,
        repeat=5,
    )

    validation_dataset = batch_dataset(
        val,
        p_batch_size,
        i_max_bp,
        shuffle=False,
        repeat=1,
    )

    model = regressor(
        p_batch_size,
        p_lr,
        p_dropout,
        p_ff_d_model,
        p_pca_heads,
        1,
        1024,
        p_token_dim,
        p_ff_clr,
        p_enc_layers,
        p_enc_heads,
        32,
        i_max_bp,
        mean,
        std
    )

    for x, y in validation_dataset.take(1):
        model(x)
    model.summary()

    reg_out_callbacks = [
        MAE_Scatter(
            'training',
            validation_dataset,
            metadata[metadata.index.isin(training_ids)],
            i_metadata_col,
            None,
            None,
            mean,
            std,
            figure_path,
            report_back_after=p_report_back_after
        )
    ]

    core_callbacks = [
        # tboard_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            "loss",
            factor=0.5,
            patients=20,
            min_lr=0.0001
        ),
        tf.keras.callbacks.EarlyStopping(
            'loss',
            patience=500
        ),
        SaveModel(
            p_output_dir,
            p_report_back_after
        ),
    ]
    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        callbacks=[
            *reg_out_callbacks,
            *core_callbacks
        ],
        epochs=p_epochs
    )


@cli.command()
@click.option(
    '--i-table-path',
    required=True,
    help=desc.TABLE_DESC,
    type=click.Path(exists=True)
)
@click.option(
    '--i-tree-path',
    required=True,
    help=desc.TABLE_DESC,
    type=click.Path(exists=True)
)
@click.option(
    '--i-max-bp',
    required=True,
    type=int
)
@click.option(
    '--p-missing-samples',
    default='error',
    type=click.Choice(['error', 'ignore'], case_sensitive=False),
    help=desc.MISSING_SAMPLES_DESC
)
@click.option(
    '--p-batch-size',
    default=8,
    show_default=True,
    type=int
)
@click.option(
    '--p-epochs',
    default=1000,
    show_default=True,
    type=int
)
@click.option(
    '--p-repeat',
    default=5,
    show_default=True,
    type=int
)
@click.option(
    '--p-dropout',
    default=0.01,
    show_default=True,
    type=float
)
@click.option(
    '--p-token-dim',
    default=512,
    show_default=True,
    type=int
)
@click.option(
    '--p-feature-attention-method',
    default='add_features',
    type=click.Choice(aam_globals['feature-attention-methods'])
)
@click.option(
    '--p-features-to-add-rate',
    default=1.,
    show_default=True,
    type=float
)
@click.option(
    '--p-ff-d-model',
    default=128,
    show_default=True,
    type=int
)
@click.option(
    '--p-ff-clr',
    default=64,
    show_default=True,
    type=int
)
@click.option(
    '--p-pca-heads',
    default=8,
    show_default=True,
    type=int
)
@click.option(
    '--p-enc-layers',
    default=2,
    show_default=True,
    type=int
)
@click.option(
    '--p-enc-heads',
    default=8,
    show_default=True,
    type=int
)
@click.option(
    '--p-include-random',
    default=True,
    show_default=True,
    type=bool
)
@click.option(
    '--p-lr',
    default=0.01,
    show_default=True,
    type=float
)
@click.option(
    '--p-report-back-after',
    default=5,
    show_default=True,
    type=int
)
@click.option(
    '--p-output-dir',
    required=True
)
def unifrac_regressor(
    i_table_path: str,
    i_tree_path: str,
    i_max_bp: int,
    p_missing_samples: str,
    p_batch_size: int,
    p_epochs: int,
    p_repeat: int,
    p_dropout: float,
    p_token_dim: int,
    p_feature_attention_method: str,
    p_features_to_add_rate: float,
    p_ff_d_model: int,
    p_ff_clr: int,
    p_pca_heads: int,
    p_enc_layers: int,
    p_enc_heads: int,
    p_include_random: bool,
    p_lr: float,
    p_report_back_after: int,
    p_output_dir: str
):
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)

    figure_path = os.path.join(p_output_dir, 'figures')
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # table = shuffle_table(load_biom_table(i_table_path))

    # feature_dataset = get_sequencing_dataset(table)
    # feature_count_dataset = get_sequencing_count_dataset(table)

    # regression_dataset = get_unifrac_dataset(
    #     i_table_path,
    #     i_tree_path,
    # )
    # mean, std = 0., 1.
    # full_dataset, sequence_tokenizer = combine_count_datasets(
    #     feature_dataset,
    #     feature_count_dataset,
    #     regression_dataset,
    #     i_max_bp,
    #     add_index=True
    # )
    # training, val = train_val_split(
    #     full_dataset,
    #     train_percent=.9
    # )
    # training_dataset = batch_dataset(
    #     training,
    #     p_batch_size,
    #     i_max_bp,
    #     shuffle=True,
    #     repeat=p_repeat,
    #     dist=True
    # )

    # validation_dataset = batch_dataset(
    #     val,
    #     p_batch_size,
    #     i_max_bp,
    #     shuffle=False,
    #     repeat=1,
    #     dist=True
    # )

    # model = pretrain_unifrac(
    #     p_batch_size,
    #     p_lr,
    #     p_dropout,
    #     p_ff_d_model,
    #     p_pca_heads,
    #     1,
    #     1024,
    #     p_token_dim,
    #     p_ff_clr,
    #     p_enc_layers,
    #     p_enc_heads,
    #     32,
    #     i_max_bp,
    #     mean,
    #     std,
    #     include_count=False,
    #     include_random=p_include_random,
    #     sequence_tokenizer=sequence_tokenizer
    # )

    # for x, y in validation_dataset.take(1):
    #     ind, seq, rclr = x
    #     model(x)
    # model.summary()

    table = load_biom_table(i_table_path)
    o_ids = tf.constant(table.ids(axis='observation'))
    table = table.transpose()
    data = table.matrix_data.tocoo()
    row_ind = data.row
    col_ind = data.col
    values = data.data
    indices = [[r, c] for r, c in zip(row_ind, col_ind)]
    table_data = tf.sparse.SparseTensor(indices=indices, values=values,
                                        dense_shape=table.shape)
    table_data = tf.sparse.reorder(table_data)
    table_dataset = tf.data.Dataset.from_tensor_slices(table_data)
    sequence_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=8,
        split='character',
        output_mode='int',
        output_sequence_length=i_max_bp
    )
    sequence_tokenizer.adapt(o_ids[:10])
    unifrac_dataset = get_unifrac_dataset(i_table_path, i_tree_path)

    dataset = (
        tf.data.Dataset.zip(
            tf.data.Dataset.range(unifrac_dataset.cardinality()),
            table_dataset,
            unifrac_dataset
        )
        .shuffle(unifrac_dataset.cardinality(), reshuffle_each_iteration=True)
        .batch(8, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    )

    mean = 0
    std = 1

    model = pretrain_unifrac(
        p_batch_size,
        p_lr,
        p_dropout,
        p_ff_d_model,
        p_pca_heads,
        1,
        1024,
        p_token_dim,
        p_ff_clr,
        p_enc_layers,
        p_enc_heads,
        32,
        i_max_bp,
        mean,
        std,
        include_count=False,
        include_random=p_include_random,
        o_ids=o_ids,
        sequence_tokenizer=sequence_tokenizer
    )

    for data in dataset.take(1):
        inputs, outputs = model._extract_data(data)
        model(inputs)
    model.summary()

    core_callbacks = [
        # tboard_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            "loss",
            factor=0.8,
            patients=5,
            min_lr=0.00001
        ),
        tf.keras.callbacks.EarlyStopping(
            'loss',
            patience=500
        ),
        SaveModel(
            p_output_dir,
            p_report_back_after
        ),
    ]
    model.fit(
        dataset,
        # validation_data=validation_dataset,
        callbacks=[
            *core_callbacks
        ],
        epochs=p_epochs
    )


@cli.command()
@click.option(
    '--i-table-path',
    required=True,
    help=desc.TABLE_DESC,
    type=click.Path(exists=True)
)
@click.option(
    '--i-metadata-path',
    required=True,
    help=desc.TABLE_DESC,
    type=click.Path(exists=True)
)
@click.option(
    '--i-metadata-col',
    required=True,
    type=str
)
@click.option(
    '--i-max-bp',
    required=True,
    type=int
)
@click.option(
    '--p-missing-samples',
    default='error',
    type=click.Choice(['error', 'ignore'], case_sensitive=False),
    help=desc.MISSING_SAMPLES_DESC
)
@click.option(
    '--p-batch-size',
    default=8,
    show_default=True,
    type=int
)
@click.option(
    '--p-epochs',
    default=1000,
    show_default=True,
    type=int
)
@click.option(
    '--p-repeat',
    default=5,
    show_default=True,
    type=int
)
@click.option(
    '--p-dropout',
    default=0.01,
    show_default=True,
    type=float
)
@click.option(
    '--p-token-dim',
    default=512,
    show_default=True,
    type=int
)
@click.option(
    '--p-feature-attention-method',
    default='add_features',
    type=click.Choice(aam_globals['feature-attention-methods'])
)
@click.option(
    '--p-features-to-add-rate',
    default=1.,
    show_default=True,
    type=float
)
@click.option(
    '--p-ff-d-model',
    default=128,
    show_default=True,
    type=int
)
@click.option(
    '--p-ff-clr',
    default=64,
    show_default=True,
    type=int
)
@click.option(
    '--p-pca-heads',
    default=8,
    show_default=True,
    type=int
)
@click.option(
    '--p-enc-layers',
    default=2,
    show_default=True,
    type=int
)
@click.option(
    '--p-enc-heads',
    default=8,
    show_default=True,
    type=int
)
@click.option(
    '--p-include-random',
    default=True,
    show_default=True,
    type=bool
)
@click.option(
    '--p-lr',
    default=0.01,
    show_default=True,
    type=float
)
@click.option(
    '--p-report-back-after',
    default=5,
    show_default=True,
    type=int
)
@click.option(
    '--p-base-model-path',
    required=True,
    type=click.Path(exists=True)
)
@click.option(
    '--p-output-dir',
    required=True
)
def transfer_learn_fit_regressor(
    i_table_path: str,
    i_metadata_path: str,
    i_metadata_col: str,
    i_max_bp: int,
    p_missing_samples: str,
    p_batch_size: int,
    p_epochs: int,
    p_repeat: int,
    p_dropout: float,
    p_token_dim: int,
    p_feature_attention_method: str,
    p_features_to_add_rate: float,
    p_ff_d_model: int,
    p_ff_clr: int,
    p_pca_heads: int,
    p_enc_layers: int,
    p_enc_heads: int,
    p_include_random: bool,
    p_lr: float,
    p_report_back_after: int,
    p_base_model_path: str,
    p_output_dir: str
):
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)

    figure_path = os.path.join(p_output_dir, 'figures')
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    table = shuffle_table(load_biom_table(i_table_path))
    metadata = pd.read_csv(
        i_metadata_path, sep='\t',
        index_col=0
    )

    # check for mismatch samples
    ids = table.ids(axis='sample')
    shared_ids = set(ids).intersection(set(metadata.index))
    min_ids = min(len(shared_ids), len(ids), len(metadata.index))
    max_ids = max(len(shared_ids), len(ids), len(metadata.index))
    if len(shared_ids) == 0:
        raise Exception('Table and Metadata have no matching sample ids')
    if min_ids != max_ids and p_missing_samples == 'error':
        raise Exception('Table and Metadata do share all same sample ids.')
    elif min_ids != max_ids and p_missing_samples == 'ignore':
        print('Warning: Table and Metadata do share all same sample ids.')
        print('Table and metadata will be filtered')
        table = table.filter(shared_ids, inplace=False)
        metadata = metadata[metadata.index.isin(shared_ids)]
        ids = table.ids(axis='sample')

    # TODO: check for invalid metadata values
    table = table.remove_empty(axis='observation', inplace=False)
    feature_dataset = get_sequencing_dataset(table)
    feature_count_dataset = get_sequencing_count_dataset(table)
    metadata = filter_and_reorder(metadata, ids)
    regression_data = extract_col(
        metadata,
        i_metadata_col,
        output_dtype=np.float32
    )
    regression_dataset, mean, std = convert_to_normalized_dataset(
        regression_data,
        'z'
    )
    full_dataset = combine_count_datasets(
        feature_dataset,
        feature_count_dataset,
        regression_dataset,
        i_max_bp,
        add_index=True,
        return_tokenizer=False
    )
    training, val = train_val_split(
        full_dataset,
        train_percent=.8
    )
    training_size = training.cardinality().numpy()
    training_ids = ids[:training_size]
    training_dataset = batch_dataset(
        training,
        p_batch_size,
        i_max_bp,
        shuffle=True,
        repeat=5,
    )

    validation_dataset = batch_dataset(
        val,
        p_batch_size,
        i_max_bp,
        shuffle=False,
        repeat=1,
    )
    base_model = tf.keras.models.load_model(p_base_model_path)
    base_model.trainable = False

    model = TransferLearnNucleotideModel(
        base_model,
        mean,
        std,
        include_random=p_include_random,
        include_count=True
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=p_lr,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    model.compile(
        optimizer=optimizer,
        jit_compile=False
    )

    for x, y in validation_dataset.take(1):
        base_model(x)
        model(x)
    model.summary()

    reg_out_callbacks = [
        MAE_Scatter(
            'training',
            validation_dataset,
            metadata[metadata.index.isin(training_ids)],
            i_metadata_col,
            None,
            None,
            mean,
            std,
            figure_path,
            report_back_after=p_report_back_after
        )
    ]

    core_callbacks = [
        # tboard_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            "loss",
            factor=0.8,
            patients=10,
            min_lr=0.0001
        ),
        tf.keras.callbacks.EarlyStopping(
            'loss',
            patience=500
        ),
        SaveModel(
            p_output_dir,
            p_report_back_after
        ),
    ]
    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        callbacks=[
            *reg_out_callbacks,
            *core_callbacks
        ],
        epochs=p_epochs
    )


@cli.command()
@click.option(
    '--i-table-path',
    required=True,
    help=desc.TABLE_DESC,
    type=click.Path(exists=True)
)
@click.option(
    '--i-model-path',
    required=True,
    type=click.Path(exists=True)
)
@click.option(
    '--m-metadata-file',
    required=True,
    type=click.Path(exists=True)
)
@click.option(
    '--m-metadata-column',
    required=True,
    help=desc.METADATA_COL_DESC,
    type=str
)
@click.option(
    '--p-normalize',
    default='minmax',
    type=click.Choice(['minmax', 'z', 'none'])
)
@click.option(
    '--p-missing-samples',
    default='error',
    type=click.Choice(['error', 'ignore'], case_sensitive=False),
    help=desc.MISSING_SAMPLES_DESC
)
@click.option(
    '--p-output-dir',
    required=True
)
def scatter_plot(
    i_table_path,
    i_model_path,
    m_metadata_file,
    m_metadata_column,
    p_normalize,
    p_missing_samples,
    p_output_dir
):
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)
    model = tf.keras.models.load_model(i_model_path)
    dataset_obj = _create_dataset(
        i_table_path,
        m_metadata_file,
        m_metadata_column,
        p_normalize,
        p_missing_samples
    )
    training = dataset_obj['dataset']
    training_no_shuffle = batch_dataset(
        training,
        32,
        shuffle=False
    )

    mean = dataset_obj['mean']
    std = dataset_obj['std']
    std = model.std
    mean = model.mean
    output = model.predict(training_no_shuffle)
    pred_val = tf.concat(output['regression'], axis=0)
    pred_val = tf.squeeze(pred_val)
    pred_val = pred_val*std + mean
    true_val = tf.concat(
        [y["reg_out"] for _, y in training_no_shuffle],
        axis=0
    )
    true_val = tf.squeeze(true_val)
    true_val = true_val*std + mean
    mae = tf.reduce_mean(tf.abs(true_val - pred_val))

    min_x = tf.reduce_min(true_val)
    max_x = tf.reduce_max(true_val)
    coeff = np.polyfit(true_val, pred_val, deg=1)
    p = np.poly1d(coeff)
    xx = np.linspace(min_x, max_x, 50)
    yy = p(xx)

    diag = np.polyfit(true_val, true_val, deg=1)
    p = np.poly1d(diag)
    diag_xx = np.linspace(min_x, max_x, 50)
    diag_yy = p(diag_xx)
    data = {
        "#SampleID": dataset_obj["sample_ids"],
        "pred": pred_val.numpy(),
        "true": true_val.numpy()
    }
    data = pd.DataFrame(data=data)
    plot = sns.scatterplot(data, x="true", y="pred")
    plt.plot(xx, yy)
    plt.plot(diag_xx, diag_yy)
    mae = '%.4g' % mae
    plot.set(xlabel='True')
    plot.set(ylabel='Predicted')
    plot.set(title=f"Mean Absolute Error: {mae}")
    plt.savefig(
        os.path.join(p_output_dir, 'scatter-plot.png'),
        bbox_inches="tight"
    )
    plt.close()
    data["residual"] = data["true"] - data["pred"]

    mean_residual = np.mean(np.abs(data["residual"]))
    mean_residual = '%.4g' % mean_residual
    plot = sns.displot(data, x="residual")
    plot.set(title=f"Mean Absolute Residual: {mean_residual}")
    plt.savefig(
        os.path.join(p_output_dir, 'residual-plot.png'),
        bbox_inches="tight"
    )
    plt.close()
    data.to_csv(
        os.path.join(p_output_dir, 'sample-residuals.tsv'),
        sep='\t',
        index=False
    )


def main():
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    cli()


if __name__ == '__main__':
    main()
