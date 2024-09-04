"""gotu.py"""

import json
import os
from datetime import datetime

import biom
import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import aam._parameter_descriptions as desc
from aam.callbacks import SaveModel
from aam.data_utils import load_data
from aam.utils import LRDecrease
from attention_regression.callbacks import MAE_Scatter
from gotu.gotu_model import GOTUModel


@click.group()
def cli():
    pass


def _aam_globals():
    return {"feature-attention-methods": ["add_features", "mask_features", "none"]}


aam_globals = _aam_globals()


@cli.command()
@click.option("--i-table-path", required=True, help=desc.TABLE_DESC, type=click.Path(exists=True))
@click.option(
    "--i-gotu-path",
    required=True,
    help=desc.TABLE_DESC,
    type=click.Path(exists=True),
)
@click.option("--i-max-bp", required=True, type=int)
@click.option("--p-batch-size", default=8, show_default=True, type=int)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-repeat", default=5, show_default=True, type=int)
@click.option("--p-dropout", default=0.01, show_default=True, type=float)
@click.option("--p-token-dim", default=512, show_default=True, type=int)
@click.option("--p-ff-d-model", default=128, show_default=True, type=int)
@click.option("--p-ff-clr", default=64, show_default=True, type=int)
@click.option("--p-pca-heads", default=8, show_default=True, type=int)
@click.option("--p-enc-layers", default=2, show_default=True, type=int)
@click.option("--p-enc-heads", default=8, show_default=True, type=int)
@click.option("--p-include-random", default=True, show_default=True, type=bool)
@click.option("--p-lr", default=0.01, show_default=True, type=float)
@click.option("--p-report-back-after", default=5, show_default=True, type=int)
@click.option("--p-base-model-path", required=True, type=click.Path(exists=True))
@click.option("--p-output-dir", required=True)
@click.option("--p-model-weights-path", default=None, required=False, type=click.Path(exists=True))
def sequence2sequence(
    i_table_path: str,
    i_gotu_path: str,
    i_max_bp: int,
    p_batch_size: int,
    p_epochs: int,
    p_repeat: int,
    p_dropout: float,
    p_token_dim: int,
    p_ff_d_model: int,
    p_ff_clr: int,
    p_pca_heads: int,
    p_enc_layers: int,
    p_enc_heads: int,
    p_include_random: bool,
    p_lr: float,
    p_report_back_after: int,
    p_base_model_path: str,
    p_output_dir: str,
    p_model_weights_path: str,
):
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)

    figure_path = os.path.join(p_output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    data_obj = load_data(
        i_table_path,
        i_max_bp,
        p_batch_size,
        repeat=p_repeat,
        biom_path=i_gotu_path,
        shuffle_samples=True,
        train_percent=0.8,
    )
    base_model = tf.keras.models.load_model(p_base_model_path, compile=False)
    base_model.trainable = False

    num_gotus = data_obj["num_gotus"]

    gotu_model = GOTUModel(
        base_model=base_model,
        dropout_rate=p_dropout,
        batch_size=p_batch_size,
        max_bp=i_max_bp,
        num_gotus=num_gotus,
        count_ff_dim=p_ff_clr,
        num_layers=p_enc_layers,
        num_attention_heads=p_enc_heads,
        dff=p_ff_d_model,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LRDecrease(),
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    gotu_model.compile(
        optimizer=optimizer,
        o_ids=data_obj["o_ids"],
        gotu_ids=data_obj["gotu_ids"],
        sequence_tokenizer=data_obj["sequence_tokenizer"],
        # weighted_metrics=[],
    )

    #   gotu_model.build(input_shape=[(p_batch_size, None, 150), (p_batch_size, None)])
    for data in data_obj["training_dataset"].take(1):
        x, y = gotu_model._extract_data(data)
        gotu_model(x)

    if p_model_weights_path is not None:
        gotu_model.load_weights(f"{p_model_weights_path}/best_model")
        print("!!!Loaded Model Weights!!!")

    gotu_model.summary()

    core_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.8, patience=10, min_lr=0.0001),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(p_output_dir, "best_model"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            save_format="tf",
            save_weights_only=True,
        ),
    ]

    gotu_model.fit(
        data_obj["training_dataset"],
        validation_data=data_obj["validation_dataset"],
        callbacks=core_callbacks,
        epochs=p_epochs,
    )


@cli.command()
@click.option("--i-table-path", required=True, help=desc.TABLE_DESC, type=click.Path(exists=True))
@click.option(
    "--i-gotu-path",
    required=True,
    help=desc.TABLE_DESC,
    type=click.Path(exists=True),
)
@click.option("--i-max-bp", required=True, type=int)
@click.option("--p-batch-size", default=8, show_default=True, type=int)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-repeat", default=5, show_default=True, type=int)
@click.option("--p-dropout", default=0.01, show_default=True, type=float)
@click.option("--p-token-dim", default=512, show_default=True, type=int)
@click.option("--p-ff-d-model", default=128, show_default=True, type=int)
@click.option("--p-ff-clr", default=64, show_default=True, type=int)
@click.option("--p-pca-heads", default=8, show_default=True, type=int)
@click.option("--p-enc-layers", default=2, show_default=True, type=int)
@click.option("--p-enc-heads", default=8, show_default=True, type=int)
@click.option("--p-include-random", default=True, show_default=True, type=bool)
@click.option("--p-lr", default=0.01, show_default=True, type=float)
@click.option("--p-report-back-after", default=5, show_default=True, type=int)
@click.option("--p-base-model-path", required=True, type=click.Path(exists=True))
@click.option("--p-output-dir", required=True)
@click.option("--p-model-weights-path", default=None, required=True, type=click.Path(exists=True))
def predict_s2s(
    i_table_path: str,
    i_gotu_path: str,
    i_max_bp: int,
    p_batch_size: int,
    p_epochs: int,
    p_repeat: int,
    p_dropout: float,
    p_token_dim: int,
    p_ff_d_model: int,
    p_ff_clr: int,
    p_pca_heads: int,
    p_enc_layers: int,
    p_enc_heads: int,
    p_include_random: bool,
    p_lr: float,
    p_report_back_after: int,
    p_base_model_path: str,
    p_output_dir: str,
    p_model_weights_path: str,
):
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)

    figure_path = os.path.join(p_output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    data_obj = load_data(
        i_table_path,
        i_max_bp,
        p_batch_size,
        repeat=p_repeat,
        biom_path=i_gotu_path,
        shuffle_samples=False,
        train_percent=1.0,
    )
    base_model = tf.keras.models.load_model(p_base_model_path, compile=False)
    base_model.trainable = False

    num_gotus = data_obj["num_gotus"]

    gotu_model = GOTUModel(
        base_model=base_model,
        dropout_rate=p_dropout,
        batch_size=p_batch_size,
        max_bp=i_max_bp,
        num_gotus=num_gotus,
        count_ff_dim=p_ff_clr,
        num_layers=p_enc_layers,
        num_attention_heads=p_enc_heads,
        dff=p_ff_d_model,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LRDecrease(),
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    gotu_model.compile(
        optimizer=optimizer,
        o_ids=data_obj["o_ids"],
        gotu_ids=data_obj["gotu_ids"],
        sequence_tokenizer=data_obj["sequence_tokenizer"],
        weighted_metrics=[],
    )
    if p_model_weights_path is not None:
        gotu_model.load_weights(f"{p_model_weights_path}/best_model").expect_partial()
        tf.print("!!!!!!!!!!LOADED WEIGHTS!!!!!!!!!!!!!!")

    gotu_model.trainable = False

    for data in data_obj["training_dataset"].take(1):
        x, y = gotu_model._extract_data(data)
        gotu_model(x)

    # for layer in gotu_model.layers:  # eventually setup as load test
    #     tf.print(layer.get_weights())

    gotu_model.summary()
    
    with open("/home/jokirkland/data/asv2gotu/paired_asv_gotu_data/gotu_dict.json", "r") as file:
        gotu_dict = json.load(file)
    gotu_dict = {int(key): value for key, value in gotu_dict.items()}
    col_index = 0
    pred_biom_data = np.zeros((len(gotu_dict), len(data_obj["sample_ids"])))
    gotu_obv_dict = {}
    for i in range(len(gotu_dict)):
        gotu_obv_dict[gotu_dict[i]] = i

    for data in tqdm(data_obj["training_dataset"], desc="Processing data"):
        x, y = gotu_model._extract_data(data)
        result = gotu_model(x)
        y_pred = tf.argmax(result[0][0], axis=-1)

        for i in range(len(y_pred)):
            for j in range(len(y_pred[i, :])):
                gotu_code = y_pred[i, :][j]
                if gotu_code:
                    pred_biom_data[gotu_code, col_index] = 1
            col_index += 1

    pred_biom = biom.table.Table(pred_biom_data, list(gotu_dict.values()), data_obj["sample_ids"])
    with biom.util.biom_open("/home/jokirkland/data/asv2gotu/aam_testing/predictions/test_pred.biom", "w") as f:
        pred_biom.to_hdf5(f, "Predicted GOTUs Using DeepLearning")

    # prediction_output_path = f"{p_output_dir}/test_pred.csv"
    # tf.print(prediction_output_path)
    # tf.print(predictions)
    # tf.print(predictions, output_stream=f"{prediction_output_path}")
    # tf.print(f"Predictions saved to {prediction_output_path}")


def main():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    cli()


if __name__ == "__main__":
    main()
