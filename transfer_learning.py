import click
import tensorflow as tf
import aam._parameter_descriptions as desc
from aam.callbacks import ProjectEncoder
from aam.data_utils import (
    get_sequencing_dataset, get_unifrac_dataset, combine_datasets,
    batch_dataset,
)
from aam.model_utils import pretrain_unifrac
from aam.cli_util import aam_model_options


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
            dff,
            d_model,
            enc_layers,
            enc_heads,
            output_dim,
            lr,
            output_dir):
    seq_dataset = get_sequencing_dataset(i_table)
    unifrac_dataset = get_unifrac_dataset(i_table, i_tree)
    dataset = combine_datasets(seq_dataset,
                               unifrac_dataset,
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
                             dff,
                             d_model,
                             enc_layers,
                             enc_heads,
                             output_dim)
    model.summary()

    model.fit(training_dataset,
              validation_data=validation_dataset,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[ProjectEncoder(i_table,
                                        i_tree,
                                        output_dir,
                                        batch_size)])


def main():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    transfer_learning(prog_name='transfer_learning')


if __name__ == '__main__':
    main()
