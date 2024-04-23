"""gotu.py"""

import click
import biom
import numpy as np
import tensorflow as tf

from aam.callbacks import SaveModel
from gotu.gotu_data_utils import (
    create_training_dataset,
    create_prediction_dataset,
)
from gotu.gotu_model import gotu_classification, gotu_predict


def run_gotu_training(asv_fp: str, gotu_fp: str, **kwargs) -> None:
    model_fp = kwargs.get("model_fp", None)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus),
                "Physical GPUs,",
                len(logical_gpus),
                "Emotional Support GPUs",
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    training_dataset, validation_dataset, gotu_dict = create_training_dataset(
        gotu_fp, asv_fp
    )

    if model_fp is None:
        model = gotu_classification(
            batch_size=8,
            dropout=0.1,
            dff=256,
            d_model=128,
            enc_layers=4,
            enc_heads=8,
            max_bp=150,
        )
    model.summary()
    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        callbacks=[SaveModel("gotu_decoder_model")],
        epochs=1000,
    )


def run_gotu_predictions(
    asv_fp: str, gotu_fp: str, model_fp: str, pred_out_path: str
) -> None:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus),
                "Physical GPUs,",
                len(logical_gpus),
                "Emotional Support GPUs",
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    asv_biom = biom.load_table(asv_fp)
    gotu_biom = biom.load_table(gotu_fp)
    asv_samples = asv_biom.ids(axis="sample")
    gotu_ids = gotu_biom.ids(axis="observation")

    pred_biom_data = np.zeros((len(gotu_ids), len(asv_samples)))
    gotu_obv_dict = {}
    for i in range(len(gotu_ids)):
        gotu_obv_dict[gotu_ids[i]] = i

    combined_dataset, gotu_dict = create_prediction_dataset(
        gotu_fp, asv_fp, 8
    )
    model = gotu_predict(
        8,
        model_fp,
        dropout=0,
        dff=512,
        d_model=128,
        enc_layers=4,
        enc_heads=8,
        max_bp=150,
    )
    model.summary()

    col_index = 0
    for x, y in combined_dataset:
        predictions = model.predict_on_batch(x)
        predicted_classes = np.argmax(predictions, axis=-1)

        y_pred = predicted_classes
        for i in range(len(y_pred)):
            for j in range(len(y_pred[i, :])):
                gotu_code = y_pred[i, :][j]
                if gotu_code > 3:
                    gotu_name = gotu_dict[int(gotu_code)]
                    gotu_pos = gotu_obv_dict[gotu_name]
                    pred_biom_data[gotu_pos, col_index] = 1
            col_index += 1

    pred_biom = biom.table.Table(pred_biom_data, gotu_ids, asv_samples)
    with biom.util.biom_open(f"{pred_out_path}", "w") as f:
        pred_biom.to_hdf5(f, "Predicted GOTUs Using DeepLearning")


@click.command()
@click.argument("asv_fp", type=click.Path(exists=True))
@click.argument("gotu_fp", type=click.Path(exists=True))
@click.option(
    "--load_model", type=click.Path(), help="Load previously trained model"
)
@click.option(
    "--model_fp", type=click.Path(), help="File path to save the model."
)
def run_gotu_training_cli(asv_fp, gotu_fp, model_fp):
    run_gotu_training(asv_fp, gotu_fp, model_fp=model_fp)


@click.command()
@click.argument("asv_fp", type=click.Path(exists=True))
@click.argument("gotu_fp", type=click.Path(exists=True))
@click.argument("model_fp", type=click.Path(exists=True))
@click.argument("pred_out_path", type=click.Path())
def run_gotu_predictions_cli(asv_fp, gotu_fp, model_fp, pred_out_path):
    run_gotu_predictions(asv_fp, gotu_fp, model_fp, pred_out_path)


@click.group()
def cli():
    pass


cli.add_command(run_gotu_training_cli, "train")
cli.add_command(run_gotu_predictions_cli, "predict")

if __name__ == "__main__":
    cli()
