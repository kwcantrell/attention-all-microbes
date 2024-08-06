"""gotu.py"""

import os
from datetime import datetime

import click
import tensorflow as tf

import aam._parameter_descriptions as desc
from aam.callbacks import SaveModel
from aam.data_utils import load_data
from aam.utils import LRDecrease
from attention_regression.callbacks import MAE_Scatter
from gotu.gotu_model import GOTUModel


@click.group()
class cli:
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
@click.option(
    "--p-missing-samples",
    default="error",
    type=click.Choice(["error", "ignore"], case_sensitive=False),
    help=desc.MISSING_SAMPLES_DESC,
)
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
def sequence2sequence(
    i_table_path: str,
    i_max_bp: int,
    p_missing_samples: str,
    p_batch_size: int,
    p_epochs: int,
    p_repeat: int,
    p_dropout: float,
    p_token_dim: int,
    p_feature_attention_method: str,
    p_features_to_add_rate: float,
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
    p_tensorboard: bool,
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
        shuffle_samples=True,
        missing_samples_flag=p_missing_samples,
        train_percent=0.8,
    )

    train_size = data_obj["training_dataset"].cardinality().numpy()

    base_model = tf.keras.models.load_model(p_base_model_path, compile=False)
    base_model = tf.keras.Model(
        inputs=base_model.get_layer("nucleotide_embedding").input,
        outputs=base_model.get_layer("nucleotide_embedding").output,
        name="unifrac_model",
    )
    base_model.trainable = False

    num_gotus = data_obj["num_gotus"]

    gotu_model = GOTUModel(
        base_model=base_model,
        dropout_rate=p_dropout,
        batch_size=p_batch_size,
        max_bp=i_max_bp,
        num_gotus=num_gotus,
        pca_hidden_dim=base_model.pca_hidden_dim,
        pca_heads=p_pca_heads,
        pca_layers=base_model.pca_layers,
        count_ff_dim=p_ff_clr,
        num_layers=p_enc_layers,
        num_attention_heads=p_enc_heads,
        dff=p_ff_d_model,
        d_model=p_token_dim,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LRDecrease(),
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )

    gotu_model.compile(
        optimizer=optimizer,
        loss=gotu_model.regresssion_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    gotu_model.build(input_shape=[(p_batch_size, None, 150), (p_batch_size, None, 1)])
    gotu_model.summary()

    # core_callbacks = [
    #     tf.keras.callbacks.ReduceLROnPlateau(
    #         monitor="loss", factor=0.8, patience=10, min_lr=0.0001
    #     ),
    #     tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20),
    #     tf.keras.callbacks.ModelCheckpoint(
    #         filepath=os.path.join(p_output_dir, "best_model.h5"),
    #         save_best_only=True,
    #         monitor="val_loss",
    #         mode="min",
    #     ),
    # ]

    # gotu_model.fit(
    #     data_obj["training_dataset"],
    #     validation_data=data_obj["validation_dataset"],
    #     callbacks=core_callbacks,
    #     epochs=p_epochs,
    # )


def main():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Equivalent to the two lines above

    cli()


if __name__ == "__main__":
    main()
