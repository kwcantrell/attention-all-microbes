"""gotu.py"""

import json
import os
from datetime import datetime

import aam._parameter_descriptions as desc
import biom
import click
import numpy as np
import tensorflow as tf
from aam.callbacks import SaveModel
from aam.data_utils import load_data
from aam.utils import LRDecrease, float_mask
from attention_regression.callbacks import MAE_Scatter
from gotu.translator import Translator
from gotu.gotu_model import GOTUModel
from tqdm import tqdm


@click.group()
def cli():
    pass


def _aam_globals():
    return {"feature-attention-methods": ["add_features", "mask_features", "none"]}


aam_globals = _aam_globals()


@cli.command()
@click.option(
    "--i-table-path", required=True, help=desc.TABLE_DESC, type=click.Path(exists=True)
)
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
@click.option(
    "--p-gotu-model-path", default=None, required=False, type=click.Path(exists=True)
)
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
    p_gotu_model_path: str,
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
    if p_gotu_model_path is not None:
        gotu_model = tf.keras.models.load_model(p_gotu_model_path, compile=False)

    else:
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
            start_token=data_obj["start_token"],
            end_token=data_obj["end_token"],
        )
    optimizer = tf.keras.optimizers.Adam(
        # learning_rate=LRDecrease(lr=0.0003),
        learning_rate=0.00025,
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
        accuracy_tracker=tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    )

    for data in data_obj["training_dataset"].take(1):
        x, y = gotu_model._extract_data(data)
        gotu_model(x)

    # if p_model_weights_path is not None:
    #     gotu_model.load_model(f"{p_model_weights_path}/best_model")
    #     print("!!!Loaded Model Weights!!!")

    gotu_model.summary()

    core_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.8, patience=10, min_lr=0.0001
        ),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20),
        SaveModel(
            output_dir=p_output_dir,
            report_back=1,
        ),
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=os.path.join(p_output_dir, "best_model"),
        #     save_best_only=True,
        #     monitor="val_loss",
        #     mode="min",
        #     save_format="tf",
        #     save_weights_only=True,
        # ),
    ]

    gotu_model.fit(
        data_obj["training_dataset"],
        validation_data=data_obj["validation_dataset"],
        callbacks=core_callbacks,
        epochs=p_epochs,
    )


@cli.command()
@click.option(
    "--i-table-path", required=True, help=desc.TABLE_DESC, type=click.Path(exists=True)
)
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
@click.option("--p-output-fn", required=True, type=str)
@click.option("--p-output-dir", required=True)
@click.option(
    "--p-gotu-model-path", default=None, required=True, type=click.Path(exists=True)
)
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
    p_output_fn: str,
    p_output_dir: str,
    p_gotu_model_path: str,
):
    @tf.function
    def process_batch_loop(gotu_model, dataset, num_gotus, translator):
        target_shape = (8, num_gotus)
        col_index = tf.constant(0, dtype=tf.int32)
        batch_preds = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        print("trace")
        tf.print("subsequent executions")
        for data in dataset.take(20):
            x, y = gotu_model._extract_data(data)
            encoder_inputs, _ = x
            tf.print("Num of GOTU's in Batch", tf.shape(y))
            infer_table = tf.map_fn(
                lambda x: translator(x),
                encoder_inputs,
                fn_output_signature=tf.int64,
            )
            # tf.print("INFERRED", infer_table[0])
            tf.print("Y_PRED", tf.reduce_sum(float_mask(y), axis=-2))

            y_pred_padded = tf.pad(
                infer_table,
                [[0, 0], [0, target_shape[1] - tf.shape(infer_table)[1]]],
                constant_values=0,
            )

            batch_preds = batch_preds.write(col_index, y_pred_padded)

            tf.print("Processed batch", col_index)
            col_index += 1
        t2return = tf.reshape(batch_preds.stack(), shape=(-1, num_gotus))
        tf.print(t2return.shape)
        return t2return

    def post_process_gotu_codes(y_pred, sample_ids, gotu_dict):
        pred_biom_data = np.zeros((len(gotu_dict) - 3, len(sample_ids)))

        for i in range(len(y_pred)):
            for j in range(len(y_pred[i])):
                gotu_code = y_pred[i][j]
                if gotu_code == 0 or gotu_code == 7664 or gotu_code == 7665:
                    continue
                gotu_code -= 1
                pred_biom_data[gotu_code, i] = 1

        return pred_biom_data

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
    gotu_model = tf.keras.models.load_model(p_gotu_model_path, compile=False)
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
        start_token=data_obj["start_token"],
        end_token=data_obj["end_token"],
    )

    gotu_model.trainable = False

    for data in data_obj["training_dataset"].take(1):
        # tf.print("DATA", data)
        x, y = gotu_model._extract_data(data)
        gotu_model(x)
        # tf.print(x[0].shape)
        # tf.print(type(x[0]))
        # tf.print(x[1].shape)
        # tf.print(type(x[1]))

    gotu_model.summary()

    with open(
        "/home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/thdmi/ordered/thdmi_gotu_dict.json",
        "r",
    ) as file:
        gotu_dict = json.load(file)
    gotu_dict = {int(key): value for key, value in gotu_dict.items()}

    translator = Translator(
        gotu_model,
        data_obj["start_token"],
        data_obj["end_token"],
    )
    batch_preds = process_batch_loop(
        gotu_model,
        data_obj["training_dataset"],
        num_gotus=len(gotu_dict),
        translator=translator,
    )

    batch_preds = batch_preds.numpy()
    tf.print("finished batch_preds")
    pred_biom_data = post_process_gotu_codes(
        batch_preds, data_obj["sample_ids"], gotu_dict
    )
    pred_biom = biom.table.Table(
        pred_biom_data, list(data_obj["gotu_ids"]), data_obj["sample_ids"]
    )
    outfile_full_path = f"{p_output_dir}/{p_output_fn}.biom"
    with biom.util.biom_open(outfile_full_path, "w") as f:
        pred_biom.to_hdf5(f, "Predicted GOTUs Using DeepLearning")


def main():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    cli()


if __name__ == "__main__":
    main()
