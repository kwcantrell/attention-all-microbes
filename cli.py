import click
import tensorflow as tf
import aam._parameter_descriptions as desc
from aam.cli_util import aam_model_options
from attention_regression.data_utils import (
    load_biom_table, shuffle_table, filter_and_reorder, extract_col,
    convert_table_to_dataset, batch_dataset, convert_to_normalized_dataset,
    train_val_split
)
from attention_regression.model import _construct_model
from attention_regression.callbacks import MAE_Scatter
from aam.callbacks import SaveModel
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


@click.group()
class cli:
    pass


def _create_dataset(
    i_table_path,
    m_metadata_file,
    m_metadata_column,
    p_normalize,
    p_missing_samples
):
    table = shuffle_table(load_biom_table(i_table_path))
    metadata = pd.read_csv(
        m_metadata_file, sep='\t',
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
    feature_dataset = convert_table_to_dataset(table)
    metadata = filter_and_reorder(metadata, ids)
    regression_data = extract_col(
        metadata,
        m_metadata_column,
        output_dtype=np.float32
    )
    regression_dataset, mean, std = convert_to_normalized_dataset(
        regression_data,
        p_normalize
    )
    full_dataset = tf.data.Dataset.zip((feature_dataset, regression_dataset))
    training, _ = train_val_split(
        full_dataset,
        train_percent=1.
    )
    return {
        'dataset': training,
        'sample_ids': ids,
        'mean': mean,
        'std': std,
        'metadata': metadata,
        'table': table
    }


@cli.command()
@click.option(
    '--i-table-path',
    required=True,
    help=desc.TABLE_DESC,
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
    '--m-metadata-hue',
    default='',
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
@aam_model_options
@click.option(
    '--p-output-dir',
    required=True
)
def fit_regressor(
    i_table_path: str,
    m_metadata_file: str,
    m_metadata_column: str,
    m_metadata_hue: str,
    p_normalize: str,
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

    dataset_obj = _create_dataset(
        i_table_path,
        m_metadata_file,
        m_metadata_column,
        p_normalize,
        p_missing_samples
    )
    training = dataset_obj['dataset']
    ids = dataset_obj['sample_ids']
    training_size = training.cardinality().numpy()
    training_ids = ids[:training_size]

    training_dataset = batch_dataset(
        training,
        p_batch_size,
        repeat=p_repeat,
        shuffle=True
    )
    training_no_shuffle = batch_dataset(
        training,
        p_batch_size,
        shuffle=False
    )

    table = dataset_obj['table']
    mean = dataset_obj['mean']
    std = dataset_obj['std']
    metadata = dataset_obj['metadata']
    model = _construct_model(
        table.ids(axis='observation').tolist(),
        mean,
        std,
        p_token_dim,
        p_feature_attention_method,
        p_features_to_add_rate,
        p_dropout,
        p_ff_clr,
        p_ff_d_model,
        p_pca_heads,
        p_enc_layers,
        p_enc_heads,
        p_lr
    )
    for x, y in training_dataset.take(1):
        model((x['feature'], x['rclr']))
    model.summary()

    reg_out_callbacks = [
        MAE_Scatter(
            'training',
            training_no_shuffle,
            metadata[metadata.index.isin(training_ids)],
            m_metadata_column,
            m_metadata_hue,
            m_metadata_hue,
            mean,
            std,
            figure_path,
            report_back_after=p_report_back_after
        )
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
        SaveModel(p_output_dir)
    ]
    model.fit(
        training_dataset,
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
