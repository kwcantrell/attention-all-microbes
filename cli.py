import os

import aam._parameter_descriptions as desc
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from aam.callbacks import SaveModel
from aam.cli_util import aam_model_options
from attention_regression.data_utils import batch_dataset, load_data
from attention_regression.model import AtttentionRegression


@click.group()
class cli:
    pass


@cli.command()
@click.option(
    "--i-table-path", required=True, help=desc.TABLE_DESC, type=click.Path(exists=True)
)
@click.option("--m-metadata-file", required=True, type=click.Path(exists=True))
@click.option(
    "--m-metadata-column", required=True, help=desc.METADATA_COL_DESC, type=str
)
@click.option("--m-metadata-hue", default="", type=str)
@click.option(
    "--p-normalize", default="minmax", type=click.Choice(["minmax", "z", "none"])
)
@click.option(
    "--p-missing-samples",
    default="error",
    type=click.Choice(["error", "ignore"], case_sensitive=False),
    help=desc.MISSING_SAMPLES_DESC,
)
@aam_model_options
@click.option("--p-output-dir", required=True)
@click.option("--p-dist-meta", required=True, type=click.Path(exists=True))
def fit_regressor(
    i_table_path: str,
    m_metadata_file: str,
    m_metadata_column: str,
    m_metadata_hue: str,
    p_normalize: str,
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
    p_lr: float,
    p_report_back_after: int,
    p_output_dir: str,
    p_dist_meta: str,
):
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)

    figure_path = os.path.join(p_output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    data_obj = load_data(
        i_table_path,
        p_batch_size,
        metadata_path=m_metadata_file,
        metadata_col=m_metadata_column,
        missing_samples_flag="ignore",
        repeat=p_repeat,
        train_percent=0.8,
    )
    model = AtttentionRegression(
        p_batch_size,
        p_token_dim,
        p_ff_d_model,
        p_enc_heads,
        p_enc_layers,
        1024,
        p_dropout,
        shift=data_obj["mean"],
        scale=data_obj["std"],
        o_ids=data_obj["o_ids"],
        sequence_tokenizer=data_obj["tokenizer"],
        seq_mask_rate=0.1,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=p_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    model.compile(optimizer=optimizer, jit_compile=False)

    for data in data_obj["training_dataset"].take(1):
        inputs, _ = model._extract_data(data)
        model(inputs)
    model.summary()
    core_callbacks = [
        # tboard_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            "loss", factor=0.8, patients=5, min_lr=0.00001
        ),
        tf.keras.callbacks.EarlyStopping("loss", patience=500),
        SaveModel(p_output_dir, p_report_back_after),
    ]
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
@click.option("--i-model-path", required=True, type=click.Path(exists=True))
@click.option("--m-metadata-file", required=True, type=click.Path(exists=True))
@click.option(
    "--m-metadata-column", required=True, help=desc.METADATA_COL_DESC, type=str
)
@click.option(
    "--p-normalize", default="minmax", type=click.Choice(["minmax", "z", "none"])
)
@click.option(
    "--p-missing-samples",
    default="error",
    type=click.Choice(["error", "ignore"], case_sensitive=False),
    help=desc.MISSING_SAMPLES_DESC,
)
@click.option("--p-output-dir", required=True)
def scatter_plot(
    i_table_path,
    i_model_path,
    m_metadata_file,
    m_metadata_column,
    p_normalize,
    p_missing_samples,
    p_output_dir,
):
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)
    model = tf.keras.models.load_model(i_model_path)
    dataset_obj = _create_dataset(
        i_table_path, m_metadata_file, m_metadata_column, p_normalize, p_missing_samples
    )
    training = dataset_obj["dataset"]
    training_no_shuffle = batch_dataset(training, 32, shuffle=False)

    mean = dataset_obj["mean"]
    std = dataset_obj["std"]
    std = model.std
    mean = model.mean
    output = model.predict(training_no_shuffle)
    pred_val = tf.concat(output["regression"], axis=0)
    pred_val = tf.squeeze(pred_val)
    pred_val = pred_val * std + mean
    true_val = tf.concat([y["reg_out"] for _, y in training_no_shuffle], axis=0)
    true_val = tf.squeeze(true_val)
    true_val = true_val * std + mean
    mae = tf.reduce_mean(tf.abs(true_val - pred_val))

    min_x = tf.reduce_min(true_val)
    max_x = tf.reduce_max(true_val)
    coeff = np.polyfit(true_val, pred_val, deg=1)
    p = np.poly1d(coeff)
    xx = np.linspace(min_x, max_x, 50)
    yy = p(xx)

    diag = np.polyfit(true_val, true_val, deg=1)
    p = np.poly1d(diag)
    diag_xx = np.linspace(min_x, max_x, 50)
    diag_yy = p(diag_xx)
    data = {
        "#SampleID": dataset_obj["sample_ids"],
        "pred": pred_val.numpy(),
        "true": true_val.numpy(),
    }
    data = pd.DataFrame(data=data)
    plot = sns.scatterplot(data, x="true", y="pred")
    plt.plot(xx, yy)
    plt.plot(diag_xx, diag_yy)
    mae = "%.4g" % mae
    plot.set(xlabel="True")
    plot.set(ylabel="Predicted")
    plot.set(title=f"Mean Absolute Error: {mae}")
    plt.savefig(os.path.join(p_output_dir, "scatter-plot.png"), bbox_inches="tight")
    plt.close()
    data["residual"] = data["true"] - data["pred"]

    mean_residual = np.mean(np.abs(data["residual"]))
    mean_residual = "%.4g" % mean_residual
    plot = sns.displot(data, x="residual")
    plot.set(title=f"Mean Absolute Residual: {mean_residual}")
    plt.savefig(os.path.join(p_output_dir, "residual-plot.png"), bbox_inches="tight")
    plt.close()
    data.to_csv(
        os.path.join(p_output_dir, "sample-residuals.tsv"), sep="\t", index=False
    )


def main():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    cli()


if __name__ == "__main__":
    main()
