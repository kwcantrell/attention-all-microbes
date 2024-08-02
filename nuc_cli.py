import os
from datetime import datetime

import click
import tensorflow as tf

import aam._parameter_descriptions as desc
from aam.callbacks import SaveModel
from aam.data_utils import load_data
from aam.nuc_model import TransferLearnNucleotideModel
from aam.utils import LRDecrease
from attention_regression.callbacks import MAE_Scatter


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
@click.option(
    "--p-feature-attention-method",
    default="add_features",
    type=click.Choice(aam_globals["feature-attention-methods"]),
)
@click.option("--p-features-to-add-rate", default=1.0, show_default=True, type=float)
@click.option("--p-ff-d-model", default=128, show_default=True, type=int)
@click.option("--p-ff-clr", default=64, show_default=True, type=int)
@click.option("--p-pca-heads", default=8, show_default=True, type=int)
@click.option("--p-enc-layers", default=2, show_default=True, type=int)
@click.option("--p-enc-heads", default=8, show_default=True, type=int)
@click.option("--p-include-random", default=True, show_default=True, type=bool)
@click.option("--p-lr", default=0.01, show_default=True, type=float)
@click.option("--p-report-back-after", default=5, show_default=True, type=int)
@click.option("--p-output-dir", required=True)
def unifrac_regressor(
    i_table_path: str,
    i_tree_path: str,
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
    p_output_dir: str,
):
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)

    figure_path = os.path.join(p_output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    from aam.nuc_model import UnifracModel

    data_obj = load_data(
        i_table_path,
        i_max_bp,
        p_batch_size,
        tree_path=i_tree_path,
    )
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
        batch_size=p_batch_size,
        include_random=False,
        include_count=False,
        sequence_tokenizer=data_obj["sequence_tokenizer"],
        seq_mask_rate=p_features_to_add_rate,
    )
    optimizer = tf.keras.optimizers.Adam(
        LRDecrease(0.0003),
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        sequence_tokenizer=data_obj["sequence_tokenizer"],
        optimizer=optimizer,
        o_ids=data_obj["o_ids"],
        run_eagerly=False,
    )
    model.build()
    model.summary()
    core_callbacks = [
        SaveModel(p_output_dir, 1),
    ]
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model.fit(
        data_obj["training_dataset"],
        validation_data=data_obj["validation_dataset"],
        callbacks=[*core_callbacks],
        epochs=p_epochs,
    )


@cli.command()
@click.option(
    "--i-table-path", required=True, help=desc.TABLE_DESC, type=click.Path(exists=True)
)
@click.option(
    "--i-metadata-path",
    required=True,
    help=desc.TABLE_DESC,
    type=click.Path(exists=True),
)
@click.option("--i-metadata-col", required=True, type=str)
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
@click.option(
    "--p-feature-attention-method",
    default="add_features",
    type=click.Choice(aam_globals["feature-attention-methods"]),
)
@click.option("--p-features-to-add-rate", default=1.0, show_default=True, type=float)
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
@click.option("--p-tensorboard", default=False, show_default=True, type=bool)
def transfer_learn_fit_regressor(
    i_table_path: str,
    i_metadata_path: str,
    i_metadata_col: str,
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
        metadata_path=i_metadata_path,
        metadata_col=i_metadata_col,
        missing_samples_flag=p_missing_samples,
        train_percent=0.8,
    )

    train_size = data_obj["training_dataset"].cardinality().numpy()

    base_model = tf.keras.models.load_model(p_base_model_path, compile=False)
    batch_size = base_model.batch_size
    max_bp = base_model.max_bp
    sequence_tokenizer = base_model.sequence_tokenizer
    pca_hidden_dim = base_model.pca_hidden_dim
    pca_heads = base_model.pca_heads
    pca_layers = base_model.pca_layers
    base_model = tf.keras.Model(
        inputs=base_model.get_layer("nucleotide_embedding").input,
        outputs=base_model.get_layer("nucleotide_embedding").output,
        name="unifrac_model",
    )
    base_model.trainable = False
    d_model = 32
    model = TransferLearnNucleotideModel(
        base_model,
        p_dropout,
        batch_size=batch_size,
        max_bp=max_bp,
        pca_hidden_dim=pca_hidden_dim,
        pca_heads=pca_heads,
        pca_layers=pca_layers,
        shift=data_obj["mean"],
        scale=data_obj["std"],
        include_random=p_include_random,
        include_count=True,
        sequence_tokenizer=data_obj["sequence_tokenizer"],
        seq_mask_rate=0.01,
        d_model=d_model,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LRDecrease(),
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )
    model.compile(
        optimizer=optimizer,
        o_ids=data_obj["o_ids"],
        # sequence_tokenizer=sequence_tokenizer,
    )
    tf.print(sequence_tokenizer.get_vocabulary())
    model.build()
    model.summary()

    reg_out_callbacks = [
        MAE_Scatter(
            "training",
            data_obj["validation_dataset"],
            data_obj["metadata"][
                data_obj["metadata"].index.isin(data_obj["sample_ids"][train_size:])
            ],
            i_metadata_col,
            None,
            None,
            data_obj["mean"],
            data_obj["std"],
            figure_path,
            report_back_after=p_report_back_after,
        )
    ]

    core_callbacks = [
        # tensorboard_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            "loss", factor=0.8, patients=0, min_lr=0.0001
        ),
        tf.keras.callbacks.EarlyStopping("loss", patience=500),
        SaveModel(p_output_dir, p_report_back_after),
    ]
    if p_tensorboard:
        log_dir = p_output_dir + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            write_graph=False,
        )
        core_callbacks.append(tensorboard_callback)

    model.fit(
        data_obj["training_dataset"],
        validation_data=data_obj["validation_dataset"],
        callbacks=[*reg_out_callbacks, *core_callbacks],
        epochs=p_epochs,
    )


def main():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Equivalent to the two lines above

    cli()


if __name__ == "__main__":
    main()
