import os
from datetime import datetime

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import aam._parameter_descriptions as desc
from aam.callbacks import SaveModel
from aam.data_utils import batch_dataset, load_data
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
    print("change!!!")
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
    params = {
        "beta_1": 0.3017041123190186,
        "beta_2": 0.2696289218363902,
        "epsilon": 2.4948854833703766e-06,
        "lr": 0.0003,
    }
    model = UnifracModel(
        p_ff_d_model,
        i_max_bp,
        p_ff_d_model,
        p_pca_heads,
        8,
        p_enc_heads,
        p_enc_layers,
        2048,
        p_dropout,
        batch_size=p_batch_size,
        include_random=False,
        include_count=False,
        sequence_tokenizer=data_obj["sequence_tokenizer"],
        seq_mask_rate=p_features_to_add_rate,
    )
    from aam.utils import MyLRSchedule
    optimizer = tf.keras.optimizers.Adam(
        # MyLRSchedule()
        #tf.keras.optimizers.schedules.CosineDecayRestarts(0.0005, 2),
        LRDecrease(params["lr"]),
        # 0.0005,
        # beta_1=params["beta_1"],
        # beta_2=params["beta_2"],
        # epsilon=params["epsilon"],
    )
    # optimizer._iterations.assign(30000)
    model = tf.keras.models.load_model(
         "foundation-model/model.keras", compile=False
    )
    # optimizer.learning_rate = (model.optimizer.get_config())
    model.compile(
        sequence_tokenizer=data_obj["sequence_tokenizer"],
        optimizer=optimizer,
        o_ids=data_obj["o_ids"],
        run_eagerly=False,
    )
    model.build()
    model.summary()
    core_callbacks = [
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     "loss", factor=0.2, patients=0, min_lr=0.0000001
        # ),
        # tf.keras.callbacks.EarlyStopping("loss", patience=500),
        SaveModel(p_output_dir, 1),
    ]
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=log_dir,
    #     write_graph=True,
    #     update_freq=150,
    # )
    # core_callbacks.append(tensorboard_callback)
    model.fit(
        data_obj["training_dataset"],
        validation_data=data_obj["validation_dataset"],
        callbacks=[*core_callbacks],
        epochs=p_epochs,
    )
    # import optuna

    # from aam.nuc_model import UnifracModel

    # data_obj = load_data(
    #     i_table_path,
    #     i_max_bp,
    #     p_batch_size,
    #     tree_path=i_tree_path,
    #     shuffle_samples=True,
    # )

    # def run(optimizer, polyval=1.0):
    #     model = UnifracModel(
    #         p_ff_d_model,
    #         i_max_bp,
    #         p_ff_d_model,
    #         p_pca_heads,
    #         8,
    #         p_enc_heads,
    #         p_enc_layers,
    #         1024,
    #         p_dropout,
    #         batch_size=p_batch_size,
    #         include_random=False,
    #         include_count=False,
    #         sequence_tokenizer=data_obj["sequence_tokenizer"],
    #         seq_mask_rate=p_features_to_add_rate,
    #         polyval=polyval,
    #     )
    #     # model = tf.keras.models.load_model(
    #     #     "/home/kcantrel/amplicon-gpt/foundation-model-optuna-2/model.keras"
    #     # )
    #     model.compile(
    #         optimizer=optimizer,
    #         o_ids=data_obj["o_ids"],
    #         run_eagerly=False,
    #         polyval=polyval,
    #     )
    #     history = model.fit(data_obj["training_dataset"], epochs=20)
    #     return history

    # def get_params(trial):
    #     params = {
    #         "beta_1": trial.suggest_float("beta_1", 0.1, 0.9),
    #         "beta_2": trial.suggest_float("beta_2", 0.1, 0.99),
    #         "epsilon": trial.suggest_float("epsilon", 1e-9, 1e-5),
    #         "lr": trial.suggest_categorical("lr", [0.0001, 0.0003, 0.0005]),
    #         # "lr_type": trial.suggest_categorical("lr_type", [True, False]),
    #         # "polyval": trial.suggest_float("polyval", 0.1, 1.1),
    #     }
    #     return params

    # def get_objective():
    #     def objective(trial):
    #         params = get_params(trial)
    #         # if params["lr_type"]:
    #         #     lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
    #         #         params["lr"], 2, alpha=0.01
    #         #     )
    #         # else:
    #         lr = LRDecrease(params["lr"])
    #         optimizer = tf.keras.optimizers.Adam(
    #             # learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
    #             #     0.0005, 2, alpha=0.01
    #             # ),
    #             # LRDecrease(0.0003),
    #             lr,
    #             beta_1=params["beta_1"],
    #             beta_2=params["beta_2"],
    #             epsilon=params["epsilon"],
    #         )
    #         # history = run(optimizer=optimizer, polyval=params["polyval"])
    #         history = run(optimizer=optimizer)
    #         score = history.history["loss"][-1]
    #         score = 300000 if np.isnan(score) else score
    #         return score

    #     return objective

    # def optuna_tuner(optuna_trials=100):
    #     n_startup_trials = 20
    #     sampler = optuna.samplers.TPESampler(
    #         seed=10,
    #         n_startup_trials=n_startup_trials,
    #         consider_endpoints=True,
    #         multivariate=True,
    #     )
    #     study = optuna.create_study(sampler=sampler, direction="minimize")
    #     objective = get_objective()
    #     study.optimize(objective, n_trials=optuna_trials)
    #     trial = study.best_trial
    #     print("**" * 50 + " Finished Optimizing")
    #     print("Number of finished trials: ", len(study.trials))
    #     print("  Value: {}".format(trial.value))
    #     print("Best  Params: %s" % str(trial.params))
    #     results = trial.params.copy()
    #     return results

    # best_params = optuna_tuner(optuna_trials=200)
    # print(best_params)


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
