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
            p_batch_size,
            p_train_percent,
            p_epochs,
            p_repeat,
            p_d_model,
            p_pca_hidden_dim,
            p_pca_heads,
            p_t_heads,
            p_output_dim,
            output_dir):
    seq_dataset = get_sequencing_dataset(i_table)
    unifrac_dataset = get_unifrac_dataset(i_table, i_tree)
    dataset = combine_datasets(seq_dataset,
                               unifrac_dataset,
                               add_index=True)

    size = seq_dataset.cardinality().numpy()
    train_size = int(size*p_train_percent/p_batch_size)*p_batch_size

    training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
    training_dataset = batch_dataset(training_dataset,
                                     p_batch_size,
                                     shuffle=True,
                                     repeat=p_repeat,
                                     is_pairwise=True)

    val_data = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = batch_dataset(val_data,
                                       p_batch_size,
                                       is_pairwise=True)

    model = pretrain_unifrac(p_batch_size,
                             p_d_model,
                             p_pca_hidden_dim,
                             p_pca_heads,
                             p_t_heads,
                             p_output_dim)

    model.summary()

    model.fit(training_dataset,
              validation_data=validation_dataset,
              epochs=p_epochs,
              batch_size=p_batch_size,
              callbacks=[ProjectEncoder(i_table,
                                        i_tree,
                                        output_dir,
                                        p_batch_size)])


def main():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    transfer_learning(prog_name='transfer_learning')


if __name__ == '__main__':
    main()
