import click
import tensorflow as tf
import aam._parameter_descriptions as desc
from aam.callbacks import ProjectEncoder, SaveModel, MAE_Scatter
from aam.data_utils import (
    get_sequencing_dataset, get_unifrac_dataset, combine_datasets,
    batch_dataset, align_table_and_metadata
)
from aam.model_utils import pretrain_unifrac, transfer_regression
from aam.cli_util import aam_model_options
import os


@click.group()
def transfer_learning():
    pass


@transfer_learning.command('unifrac')
@click.option('--i-table',
              required=True,
              help=desc.TABLE_DESC,
              type=click.Path(exists=True))
@click.option('--i-tree',
              required=True,
              help=desc.TABLE_DESC,
              type=click.Path(exists=True))
@aam_model_options
@click.option('--output-dir', required=True)
def unifrac(i_table,
            i_tree,
            batch_size,
            train_percent,
            epochs,
            repeat,
            dropout,
            pca_hidden_dim,
            pca_heads,
            pca_layers,
            dff,
            d_model,
            enc_layers,
            enc_heads,
            output_dim,
            lr,
            max_bp,
            output_dir):
    seq_dataset = get_sequencing_dataset(i_table)
    unifrac_dataset = get_unifrac_dataset(i_table, i_tree)
    dataset = combine_datasets(seq_dataset,
                               unifrac_dataset,
                               max_bp,
                               add_index=True)

    size = seq_dataset.cardinality().numpy()
    train_size = int(size*train_percent/batch_size)*batch_size

    training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
    training_dataset = batch_dataset(training_dataset,
                                     batch_size,
                                     shuffle=True,
                                     repeat=repeat,
                                     is_pairwise=True)

    val_data = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = batch_dataset(val_data,
                                       batch_size,
                                       is_pairwise=True)

    model = pretrain_unifrac(batch_size,
                             lr,
                             dropout,
                             pca_hidden_dim,
                             pca_heads,
                             pca_layers,
                             dff,
                             d_model,
                             enc_layers,
                             enc_heads,
                             output_dim,
                             max_bp)

    # def scheduler(epoch, lr):
    #     if epoch <= 15:
    #         return lr
    #     return lr * tf.math.exp(-0.1)
    model.summary()
    # model.load_weights('base-model-large-ein/encoder.keras')

    model.fit(training_dataset,
              validation_data=validation_dataset,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[SaveModel(output_dir),
                         ProjectEncoder(i_table,
                                        i_tree,
                                        output_dir,
                                        batch_size)])


@transfer_learning.command()
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
def transfer_regress(i_table,
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
                     pca_layers,
                     dff,
                     d_model,
                     enc_layers,
                     enc_heads,
                     output_dim,
                     lr,
                     max_bp,
                     output_dir):
    # TODO: Normalize regress var i.e. center with a std of 0.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    model = transfer_regression(batch_size,
                                lr,
                                dropout,
                                pca_hidden_dim,
                                pca_heads,
                                pca_layers,
                                dff,
                                d_model,
                                enc_layers,
                                enc_heads,
                                output_dim,
                                max_bp)

    def scheduler(epoch, lr):
        if epoch % 10 != 0:
            return lr
        return lr * tf.math.exp(-0.1)

    model.summary()
    model.fit(training_dataset,
              validation_data=validation_dataset,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[SaveModel(output_dir),
                         MAE_Scatter(m_metadata_column,
                                     validation_dataset,
                                     output_dir),
                         tf.keras.callbacks.LearningRateScheduler(scheduler)
                         ])


def main():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    transfer_learning(prog_name='transfer_learning')


if __name__ == '__main__':
    main()
