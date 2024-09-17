import os

import aam._parameter_descriptions as desc
import click
import tensorflow as tf
from aam.callbacks import SaveModel
from aam.losses import ImbalancedCategoricalCrossEntropy, ImbalancedMSE
from aam.transfer_nuc_model import TransferLearnNucleotideModel
from aam.unifrac_data_utils import load_data
from aam.unifrac_model import UnifracModel
from attention_regression.callbacks import (
    ConfusionMatrix,
    MAE_Scatter,
)
import datetime


@click.group()
class cli:
    pass


def _aam_globals():
    return {"feature-attention-methods": ["add_features", "mask_features", "none"]}


aam_globals = _aam_globals()


@cli.command()
@click.option(
    "--i-table-path", required=True, help=desc.TABLE_DESC, type=click.Path(exists=True)
)
@click.option(
    "--i-tree-path", required=True, help=desc.TABLE_DESC, type=click.Path(exists=True)
)
@click.option("--i-max-bp", required=True, type=int)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-dropout", default=0.01, show_default=True, type=float)
@click.option("--p-ff-d-model", default=128, show_default=True, type=int)
@click.option("--p-pca-heads", default=8, show_default=True, type=int)
@click.option("--p-enc-layers", default=2, show_default=True, type=int)
@click.option("--p-enc-heads", default=8, show_default=True, type=int)
@click.option("--p-output-dir", required=True)
def fit_unifrac_regressor(
    i_table_path: str,
    i_tree_path: str,
    i_max_bp: int,
    p_epochs: int,
    p_dropout: float,
    p_ff_d_model: int,
    p_pca_heads: int,
    p_enc_layers: int,
    p_enc_heads: int,
    p_output_dir: str,
):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)

    figure_path = os.path.join(p_output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    data_obj = load_data(
        i_table_path,
        tree_path=i_tree_path,
    )

    load_model = True
    if load_model:
        model = tf.keras.models.load_model(f"{p_output_dir}/model.keras")
    else:
        model = UnifracModel(
            p_ff_d_model,
            i_max_bp,
            p_ff_d_model,
            p_pca_heads,
            8,
            p_enc_heads,
            p_enc_layers,
            1024,
            p_dropout,
        )

        optimizer = tf.keras.optimizers.Adam(0.0001)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        model.build()
        model.compile(
            optimizer=optimizer,
            run_eagerly=False,
        )
    model.summary()
    core_callbacks = [
        SaveModel(p_output_dir, 1),
    ]
    model.fit(
        data_obj["training_dataset"],
        validation_data=data_obj["validation_dataset"],
        callbacks=[*core_callbacks],
        epochs=p_epochs,
    )


@cli.command()
@click.option(
    "--i-table-path",
    required=True,
    help="Table description",
    type=click.Path(exists=True),
)
@click.option(
    "--i-metadata-path",
    required=True,
    help="Metadata description",
    type=click.Path(exists=True),
)
@click.option("--i-metadata-col", required=True, type=str)
@click.option(
    "--p-missing-samples",
    default="error",
    type=click.Choice(["error", "ignore"], case_sensitive=False),
    help="Missing samples description",
)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-report-back-after", default=5, show_default=True, type=int)
@click.option("--p-base-model-path", required=True, type=click.Path(exists=True))
@click.option("--p-output-dir", required=True)
def fit_sample_regressor(
    i_table_path: str,
    i_metadata_path: str,
    i_metadata_col: str,
    p_missing_samples: str,
    p_epochs: int,
    p_report_back_after: int,
    p_base_model_path: str,
    p_output_dir: str,
):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)

    figure_path = os.path.join(p_output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    from aam.transfer_data_utils import load_data

    load_model = False
    data_obj = load_data(
        i_table_path,
        False,
        i_metadata_path,
        i_metadata_col,
        shuffle_samples=True,
        missing_samples_flag=p_missing_samples,
    )
    if load_model:
        model = tf.keras.models.load_model("age-transfer/model.keras")
    else:
        base_model = tf.keras.models.load_model(p_base_model_path, compile=False)
        model = TransferLearnNucleotideModel(
            base_model, mean=data_obj["mean"], std=data_obj["std"]
        )
        loss = ImbalancedMSE(data_obj["max_density"])

        optimizer = tf.keras.optimizers.Adam(
            0.0001,
        )
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            run_eagerly=False,
        )
        model.build()
    model.summary()

    transfer_callbacks = [
        MAE_Scatter(
            "training",
            data_obj["validation_dataset"],
            figure_path,
            report_back_after=p_report_back_after,
        )
    ]
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(p_output_dir, log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    core_callbacks = [
        # tensorboard_callback,
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     "val_loss", factor=0.8, patients=0, min_lr=0.0001
        # ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            "val_loss", patience=25, restore_best_weights=True, start_from_epoch=100
        ),
        SaveModel(p_output_dir, p_report_back_after),
    ]

    model.fit(
        data_obj["training_dataset"],
        validation_data=data_obj["validation_dataset"],
        callbacks=[*transfer_callbacks, *core_callbacks],
        epochs=p_epochs,
    )
    model.save(os.path.join(p_output_dir, "model.keras"), save_format="keras")


@cli.command()
@click.option(
    "--i-table-path",
    required=True,
    help="Table description",
    type=click.Path(exists=True),
)
@click.option(
    "--i-metadata-path",
    required=True,
    help="Metadata description",
    type=click.Path(exists=True),
)
@click.option("--i-metadata-col", required=True, type=str)
@click.option(
    "--p-missing-samples",
    default="error",
    type=click.Choice(["error", "ignore"], case_sensitive=False),
    help="Missing samples description",
)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-report-back-after", default=5, show_default=True, type=int)
@click.option("--p-base-model-path", required=True, type=click.Path(exists=True))
@click.option("--p-output-dir", required=True)
def fit_sample_classifier(
    i_table_path: str,
    i_metadata_path: str,
    i_metadata_col: str,
    p_missing_samples: str,
    p_epochs: int,
    p_report_back_after: int,
    p_base_model_path: str,
    p_output_dir: str,
):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)

    figure_path = os.path.join(p_output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    from aam.transfer_data_utils import load_data

    load_model = False
    data_obj = load_data(
        i_table_path,
        True,
        i_metadata_path,
        i_metadata_col,
        shuffle_samples=True,
        missing_samples_flag=p_missing_samples,
    )
    if load_model:
        model = tf.keras.models.load_model("age-transfer/model.keras")
    else:
        base_model = tf.keras.models.load_model(p_base_model_path, compile=False)
        model = TransferLearnNucleotideModel(
            base_model, num_classes=data_obj["num_classes"]
        )
        loss = ImbalancedCategoricalCrossEntropy(list(data_obj["cat_counts"]))

        optimizer = tf.keras.optimizers.Adam(
            0.0001,
        )
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            run_eagerly=False,
        )
        model.build()
    model.summary()

    transfer_callbacks = [
        ConfusionMatrix(
            "Antibiotic Confusion Matrix",
            data_obj["validation_dataset"],
            data_obj["cat_labels"],
            figure_path,
            report_back_after=p_report_back_after,
        )
    ]
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(p_output_dir, log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    core_callbacks = [
        # tensorboard_callback,
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     "val_loss", factor=0.8, patients=0, min_lr=0.0001
        # ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            "val_loss", patience=25, restore_best_weights=True, start_from_epoch=100
        ),
        SaveModel(p_output_dir, p_report_back_after),
    ]

    model.fit(
        data_obj["training_dataset"],
        validation_data=data_obj["validation_dataset"],
        callbacks=[*transfer_callbacks, *core_callbacks],
        epochs=p_epochs,
    )


def main():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    cli()


if __name__ == "__main__":
    main()
