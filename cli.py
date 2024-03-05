import click
import tensorflow as tf
import aam._parameter_descriptions as desc
from aam.data_utils import (
    get_sequencing_dataset, combine_datasets,
    batch_dataset, align_table_and_metadata
)
from aam.model_utils import regression
from aam.cli_util import aam_model_options
from aam.callbacks import SaveModel, MAE_Scatter


@click.group()
class cli:
    pass


@cli.command()
@click.option('--i-table', required=True)
def classify_samples(i_feature_table):
    pass


@cli.command()
@click.option('--i-feature-table', required=True)
def confusion_matrix(i_feature_table):
    pass


@cli.command()
@click.option('--i-table',
              required=True,
              help=desc.TABLE_DESC,
              type=click.Path(exists=True))
@click.option('--m-metadata-file',
              required=True,
              type=click.Path(exists=True))
@click.option('--m-metadata-column',
              required=True,
              help=desc.METADATA_COL_DESC,
              type=str)
@click.option('--p-missing-samples',
              default='error',
              type=click.Choice(['error', 'ignore'], case_sensitive=False),
              help=desc.MISSING_SAMPLES_DESC)
@click.option('--output-dir', required=True)
def fit_classifier(i_table, m_metadata_file, metadata_column, missing_samples,
                   output_dir):
    pass


@cli.command()
@click.option('--i-table',
              required=True,
              help=desc.TABLE_DESC,
              type=click.Path(exists=True))
@click.option('--m-metadata-file',
              required=True,
              type=click.Path(exists=True))
@click.option('--m-metadata-column',
              required=True,
              help=desc.METADATA_COL_DESC,
              type=str)
@click.option('--p-missing-samples',
              default='error',
              type=click.Choice(['error', 'ignore'], case_sensitive=False),
              help=desc.MISSING_SAMPLES_DESC)
@aam_model_options
@click.option('--output-dir',
              required=True)
def fit_regressor(i_table,
                  m_metadata_file,
                  m_metadata_column,
                  p_missing_samples,
                  batch_size,
                  train_percent,
                  epochs,
                  repeat,
                  dropout,
                  pca_hidden_dim,
                  pca_heads,
                  dff,
                  d_model,
                  enc_layers,
                  enc_heads,
                  output_dim,
                  lr,
                  max_bp,
                  output_dir):
    # TODO: Normalize regress var i.e. center with a std of 0.
    table, metdata = align_table_and_metadata(i_table,
                                              m_metadata_file,
                                              m_metadata_column)
    seq_dataset = get_sequencing_dataset(table)
    y_dataset = tf.data.Dataset.from_tensor_slices(metdata[m_metadata_column])
    dataset = combine_datasets(seq_dataset,
                               y_dataset,
                               max_bp)

    size = seq_dataset.cardinality().numpy()
    train_size = int(size*train_percent/batch_size)*batch_size

    training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
    training_dataset = batch_dataset(training_dataset,
                                     batch_size,
                                     shuffle=True,
                                     repeat=repeat)

    val_data = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = batch_dataset(val_data, batch_size)

    model = regression(batch_size,
                       lr,
                       dropout,
                       pca_hidden_dim,
                       pca_heads,
                       dff,
                       d_model,
                       enc_layers,
                       enc_heads,
                       output_dim,
                       max_bp)

    model.summary()
    model.fit(training_dataset,
              validation_data=validation_dataset,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[SaveModel(output_dir),
                         MAE_Scatter(m_metadata_column,
                                     validation_dataset,
                                     output_dir)
                         ])


@cli.command()
@click.option('--i-table',
              required=True,
              help=desc.TABLE_DESC,
              type=click.Path(exists=True))
@click.option('--i-sample-estimator',
              required=True,
              help=desc.SAMPLE_CLASS_DESC)
@click.option('--output-dir', required=True)
def predict_classification(i_table, sample_estimator, output_dir):
    pass


@cli.command()
@click.option('--i-table',
              required=True,
              help=desc.TABLE_DESC,
              type=click.Path(exists=True))
@click.option('--i-sample-estimator',
              required=True,
              help=desc.SAMPLE_REGR_DESC)
@click.option('--output-dir', required=True)
def predict_regression(i_table, sample_estimator, output_dir):
    pass


@cli.command()
@click.option('--i-feature-table', required=True)
def regress_samples(i_feature_table):
    pass


@cli.command()
@click.option('--i-feature-table', required=True)
def scatterplot(i_feature_table):
    pass


@cli.command()
@click.option('--i-feature-table', required=True)
def summarize(i_feature_table):
    pass


def main():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    cli()


if __name__ == '__main__':
    main()
