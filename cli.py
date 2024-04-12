import click
import tensorflow as tf
import aam._parameter_descriptions as desc
from aam.data_utils import (
    get_sequencing_dataset, combine_datasets,
    batch_dataset, align_table_and_metadata
)
from aam.model_utils import regression, classification
from aam.cli_util import aam_model_options
from aam.callbacks import SaveModel, MAE_Scatter
import os
import pandas as pd
import numpy as np
from aam.callbacks import SaveModel
from aam.layers import PCAProjector, ReadHead
from aam.metrics import MAE
from sepsis.model import AttentionRegression, _construct_model
from sepsis.layers import FeatureEmbedding, PCA, ProjectDown, BinaryLoadings
from sepsis.callbacks import MAE_Scatter, AvgFeatureConfidence
from sepsis.data_utils import (
    load_biom_table, create_rclr_table, convert_table_to_dataset,
    convert_to_normalized_dataset, filter_and_reorder, extract_col,
    batch_dataset, shuffle_table, train_val_split
)

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
@aam_model_options
@click.option('--output-dir', required=True)
def fit_classifier(i_table,
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
                                              m_metadata_column,
                                              is_regressor=False)
    num_classes = max(metdata[m_metadata_column]) + 1
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

    model = classification(batch_size,
                           lr,
                           dropout,
                           pca_hidden_dim,
                           pca_heads,
                           pca_layers,
                           dff,
                           d_model,
                           enc_layers,
                           enc_heads,
                           num_classes,
                           max_bp)

    def scheduler(epoch, lr):
        if epoch <= 10 or epoch % 15 != 0:
            return lr
        return lr * tf.math.exp(-0.1)

    model.summary()
    model.fit(training_dataset,
              validation_data=validation_dataset,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[SaveModel(output_dir),
                         tf.keras.callbacks.LearningRateScheduler(scheduler)
                         ])


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
@click.option('--output-dir',
              required=True)
def fit_regressor(
    i_table,
    m_metadata_file,
    m_metadata_column,
    p_missing_samples,
    output_dir
):
    # TODO: Normalize regress var i.e. center with a std of 0.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    batch_size = 32
    repeat = 5
    token_dim = 512
    features_to_add = 1.0
    dropout = .10
    d_model = 1024
    ff_dim = 32
    report_back_after = 10
    epochs = 1000

    rclr_table = shuffle_table(load_biom_table(i_table))
    metadata = pd.read_csv(
        m_metadata_file, sep='\t',
        index_col=0
    )
    rclr_table, metadata = rclr_table.align_to_dataframe(metadata)
    feature_dataset = convert_table_to_dataset(rclr_table)
    ids = rclr_table.ids(axis='sample')
    # # ids = ['.'.join(id.split('.')[1:]) for id in rclr_table.ids(axis='sample')]
    metadata = filter_and_reorder(metadata, ids)
    age_in_weeks = extract_col(metadata, m_metadata_column, output_dtype=np.float32)
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

    # model = _construct_model(
    #     rclr_table.ids(axis='observation').tolist(),
    #     mean_age,
    #     std_age,
    #     token_dim,
    #     features_to_add,
    #     dropout,
    #     d_model,
    #     ff_dim,
    # )
    model = tf.keras.models.load_model('test-seassion/model/encoder.keras')
    model.feature_emb.features_add_rate = 1.0
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    model.compile(
        optimizer=optimizer
    )
    reg_out_callbacks = [
        MAE_Scatter(
            'training',
            training_no_shuffle,
            metadata[metadata.index.isin(training_ids)],
            m_metadata_column,
            None,
            'Intervention',
            mean_age,
            std_age,
            os.path.join(output_dir, 'figures'),
            report_back_after=report_back_after
        )
    ]

    emb_out_callbacks = [
        # AvgFeatureConfidence(
        #     'training',
        #     training_no_shuffle,
        #     metadata[metadata.index.isin(training_ids)],
        #     'host_age',
        #     'Age',
        #     'intervention_group',
        #     'Intervention',
        #     'sepsis/figures',
        #     report_back_after=report_back_after
        # ),
    ]

    core_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            "loss",
            factor=0.5,
            patients=0,
            min_lr=0.000001
        ),
        tf.keras.callbacks.EarlyStopping(
            'loss',
            patience=50
        ),
        SaveModel(os.path.join(output_dir, 'model'))
    ]
    model.fit(
        training_dataset,
        callbacks=[
            {"reg_out": reg_out_callbacks},
            {"emb_out": emb_out_callbacks},
            *core_callbacks,
        ],
        epochs=epochs
    )


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
