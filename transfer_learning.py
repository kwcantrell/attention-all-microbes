import click
import tensorflow as tf
import amplicon_gpt._parameter_descriptions as desc
from amplicon_gpt.callbacks import ProjectEncoder
from amplicon_gpt.data_utils import (
    get_sequencing_dataset, get_unifrac_dataset, combine_datasets,
    batch_dataset,
)
from amplicon_gpt.model_utils import transfer_learn_base, compile_model


# Allow using -h to show help information
# https://click.palletsprojects.com/en/7.x/documentation/#help-parameter-customization
CTXSETS = {"help_option_names": ["-h", "--help"]}


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
def unifrac(i_table,
            i_tree,
            batch_size,
            train_percent,
            repeat,
            epochs,
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
    validation_dataset = batch_dataset(val_data, batch_size, is_pairwise=True)

    model = transfer_learn_base(batch_size)
    # model = tf.keras.models.load_model('base-model/encoder.keras',
    #                                    safe_mode=False)
    model = compile_model(model, batch_size)

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
