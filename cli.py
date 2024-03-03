import click
import tensorflow as tf
import amplicon_gpt._parameter_descriptions as desc
from amplicon_gpt.data_utils import (
    get_sequencing_dataset, combine_datasets,
    batch_dataset, align_table_and_metadata
)
from amplicon_gpt.model_utils import regression


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
@click.option('--batch-size',
              default=8,
              type=int)
@click.option('--train-percent',
              default=0.8,
              type=float)
@click.option('--repeat',
              default=1,
              type=int)
@click.option('--epochs',
              default=100,
              type=int)
@click.option('--output-dir', required=True)
def fit_regressor(i_table,
                  m_metadata_file,
                  m_metadata_column,
                  p_missing_samples,
                  batch_size,
                  train_percent,
                  repeat,
                  epochs,
                  output_dir):
    table, metdata = align_table_and_metadata(i_table,
                                              m_metadata_file,
                                              m_metadata_column)
    seq_dataset = get_sequencing_dataset(table)
    y_dataset = tf.data.Dataset.from_tensor_slices(metdata[m_metadata_column])
    dataset = combine_datasets(seq_dataset,
                               y_dataset)

    size = seq_dataset.cardinality().numpy()
    train_size = int(size*train_percent/batch_size)*batch_size

    training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
    training_dataset = batch_dataset(training_dataset,
                                     batch_size,
                                     shuffle=True,
                                     repeat=repeat)

    val_data = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = batch_dataset(val_data, batch_size)

    model = regression(batch_size)

    model.summary()

    # Define the Keras TensorBoard callback.
    # logdir="base-model/logs/"
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
    #                                               profile_batch='50, 75',
    #   write_graph=False)
    model.fit(training_dataset,
              validation_data=validation_dataset,
              epochs=epochs,
              initial_epoch=0,
              batch_size=batch_size,
              callbacks=[])


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
