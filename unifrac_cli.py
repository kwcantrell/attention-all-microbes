import datetime
import os

import click
import pandas as pd
import tensorflow as tf
from biom import load_table
from sklearn.model_selection import KFold

import aam._parameter_descriptions as desc
from aam.callbacks import SaveModel
from aam.cv_utils import CVModel, EnsembleModel
from aam.losses import ImbalancedCategoricalCrossEntropy, ImbalancedMSE
from aam.transfer_nuc_model import TransferLearnNucleotideModel
from aam.unifrac_model import UnifracModel
from attention_regression.callbacks import (
    _confusion_matrix,
    _mean_absolute_error,
)


@click.group()
class cli:
    pass


def _aam_globals():
    return {"feature-attention-methods": ["add_features", "mask_features", "none"]}


aam_globals = _aam_globals()


@cli.command()
@click.option(
    "--i-table", required=True, help=desc.TABLE_DESC, type=click.Path(exists=True)
)
@click.option(
    "--i-tree", required=True, help=desc.TABLE_DESC, type=click.Path(exists=True)
)
@click.option("--p-max-bp", required=True, type=int)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-dropout", default=0.01, show_default=True, type=float)
@click.option("--p-ff-d-model", default=128, show_default=True, type=int)
@click.option("--p-pca-heads", default=8, show_default=True, type=int)
@click.option("--p-enc-layers", default=2, show_default=True, type=int)
@click.option("--p-enc-heads", default=8, show_default=True, type=int)
@click.option("--p-output-dir", required=True)
def fit_unifrac_regressor(
    i_table: str,
    i_tree: str,
    p_max_bp: int,
    p_epochs: int,
    p_dropout: float,
    p_ff_d_model: int,
    p_pca_heads: int,
    p_enc_layers: int,
    p_enc_heads: int,
    p_output_dir: str,
):
    from aam.unifrac_data_utils import load_data

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if not os.path.exists(p_output_dir):
        os.makedirs(p_output_dir)

    figure_path = os.path.join(p_output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    data_obj = load_data(
        i_table,
        tree_path=i_tree,
    )

    load_model = True
    if load_model:
        model = tf.keras.models.load_model(f"{p_output_dir}/model.keras")
    else:
        model = UnifracModel(
            p_ff_d_model,
            p_max_bp,
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
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(p_output_dir, log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    core_callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
        ),
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
    "--i-table",
    required=True,
    help="Table description",
    type=click.Path(exists=True),
)
@click.option("--i-base-model-path", required=True, type=click.Path(exists=True))
@click.option(
    "--m-metadata-file",
    required=True,
    help="Metadata description",
    type=click.Path(exists=True),
)
@click.option("--m-metadata-column", required=True, type=str)
@click.option(
    "--p-missing-samples",
    default="error",
    type=click.Choice(["error", "ignore"], case_sensitive=False),
    help="Missing samples description",
)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-mask-percent", default=25, show_default=True, type=int)
@click.option("--p-penalty", default=5000, type=int)
@click.option("--p-cv", default=5, type=int)
@click.option(
    "--p-test-size", default=0.2, show_default=True, type=click.FloatRange(0, 1)
)
@click.option("--output-dir", required=True, type=click.Path(exists=False))
def fit_sample_regressor(
    i_table: str,
    i_base_model_path: str,
    m_metadata_file: str,
    m_metadata_column: str,
    p_missing_samples: str,
    p_epochs: int,
    p_mask_percent: int,
    p_penalty: int,
    p_cv: int,
    p_test_size: float,
    output_dir: str,
):
    from aam.transfer_data_utils import (
        load_data,
        shuffle,
        validate_metadata,
    )

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    figure_path = os.path.join(output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    model_path = os.path.join(output_dir, "cv-models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    table = load_table(i_table)
    df = pd.read_csv(m_metadata_file, sep="\t", index_col=0)[[m_metadata_column]]
    ids, table, df = validate_metadata(table, df, p_missing_samples)
    table, df = shuffle(table, df)
    num_ids = len(ids)

    fold_indices = [i for i in range(num_ids)]
    if p_test_size > 0:
        test_size = int(num_ids * p_test_size)
        train_size = num_ids - test_size
        test_indices = fold_indices[train_size:]
        fold_indices = fold_indices[:train_size]

    print(len(test_indices), len(fold_indices))

    def _get_fold(indices, shuffle):
        fold_ids = ids[indices]
        table_fold = table.filter(fold_ids, axis="sample", inplace=False)
        df_fold = df[df.index.isin(fold_ids)]
        data = load_data(
            table_fold, False, df_fold, m_metadata_column, shuffle_samples=shuffle
        )
        return data

    kfolds = KFold(p_cv)

    models = []
    for i, (train_ind, val_ind) in enumerate(kfolds.split(fold_indices)):
        train_data = _get_fold(train_ind, shuffle=True)
        val_data = _get_fold(val_ind, shuffle=False)

        base_model = tf.keras.models.load_model(i_base_model_path, compile=False)
        model = TransferLearnNucleotideModel(
            base_model,
            mask_percent=p_mask_percent,
            shift=train_data["shift"],
            scale=train_data["scale"],
            penalty=p_penalty,
        )
        loss = ImbalancedMSE(train_data["max_density"])
        fold_label = i + 1
        model_cv = CVModel(
            model,
            train_data["dataset"],
            val_data["dataset"],
            output_dir,
            fold_label,
        )
        model_cv.fit_fold(
            loss,
            p_epochs,
            os.path.join(model_path, f"model_f{fold_label}.keras"),
            metric="mae",
        )
        models.append(model_cv)
        print(f"Fold {i+1} mae: {model_cv.metric_value}")

    best_model_path = os.path.join(output_dir, "best-model.keras")
    model_ensemble = EnsembleModel(models)
    model_ensemble.save_best_model(best_model_path)
    best_mae, ensemble_mae = model_ensemble.val_maes()
    print(
        f"Best validation mae: {best_mae}", f"Ensemble validation mae: {ensemble_mae}"
    )

    test_data = _get_fold(test_indices, shuffle=False)
    best_mae, ensemble_mae = model_ensemble.plot_fn(
        _mean_absolute_error, test_data["dataset"], figure_path
    )
    print(f"Best test mae: {best_mae}", f"Ensemble test mae: {ensemble_mae}")


@cli.command()
@click.option(
    "--i-table",
    required=True,
    help="Table description",
    type=click.Path(exists=True),
)
@click.option("--i-base-model-path", required=True, type=click.Path(exists=True))
@click.option(
    "--m-metadata-file",
    required=True,
    help="Metadata description",
    type=click.Path(exists=True),
)
@click.option("--m-metadata-column", required=True, type=str)
@click.option(
    "--p-missing-samples",
    default="error",
    type=click.Choice(["error", "ignore"], case_sensitive=False),
    help="Missing samples description",
)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-mask-percent", default=25, show_default=True, type=int)
@click.option("--p-penalty", default=5000, type=int)
@click.option("--p-cv", default=5, type=int)
@click.option(
    "--p-test-size", default=0.2, show_default=True, type=click.FloatRange(0, 1)
)
@click.option("--output-dir", required=True, type=click.Path(exists=False))
def fit_sample_classifier(
    i_table: str,
    i_base_model_path: str,
    m_metadata_file: str,
    m_metadata_column: str,
    p_missing_samples: str,
    p_epochs: int,
    p_mask_percent: int,
    p_penalty: int,
    p_cv: int,
    p_test_size: float,
    output_dir: str,
):
    from aam.transfer_data_utils import (
        load_data,
        shuffle,
        validate_metadata,
    )

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    figure_path = os.path.join(output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    model_path = os.path.join(output_dir, "cv-models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    table = load_table(i_table)
    df = pd.read_csv(m_metadata_file, sep="\t", index_col=0)[[m_metadata_column]]
    ids, table, df = validate_metadata(table, df, p_missing_samples)
    table, df = shuffle(table, df)
    num_ids = len(ids)

    fold_indices = [i for i in range(num_ids)]
    if p_test_size > 0:
        test_size = int(num_ids * p_test_size)
        train_size = num_ids - test_size
        test_indices = fold_indices[train_size:]
        fold_indices = fold_indices[:train_size]

    print(len(test_indices), len(fold_indices))

    def _get_fold(indices, shuffle):
        fold_ids = ids[indices]
        table_fold = table.filter(fold_ids, axis="sample", inplace=False)
        df_fold = df[df.index.isin(fold_ids)]
        data = load_data(
            table_fold, True, df_fold, m_metadata_column, shuffle_samples=shuffle
        )
        return data

    models = []
    kfolds = KFold(p_cv)
    for i, (train_ind, val_ind) in enumerate(kfolds.split(fold_indices)):
        train_data = _get_fold(train_ind, shuffle=True)
        val_data = _get_fold(val_ind, shuffle=False)

        base_model = tf.keras.models.load_model(i_base_model_path, compile=False)
        model = TransferLearnNucleotideModel(
            base_model,
            mask_percent=p_mask_percent,
            shift=train_data["shift"],
            scale=train_data["scale"],
            penalty=p_penalty,
            num_classes=train_data["num_classes"],
        )
        loss = ImbalancedCategoricalCrossEntropy(list(train_data["cat_counts"]))
        fold_label = i + 1
        model_cv = CVModel(
            model,
            train_data["dataset"],
            val_data["dataset"],
            output_dir,
            fold_label,
        )
        model_cv.fit_fold(
            loss,
            p_epochs,
            os.path.join(model_path, f"model_f{fold_label}.keras"),
            metric="target_loss",
        )
        models.append(model_cv)

    best_model_path = os.path.join(output_dir, "best-model.keras")
    model_ensemble = EnsembleModel(models)
    model_ensemble.save_best_model(best_model_path)
    model_ensemble.val_maes()

    test_data = _get_fold(test_indices, shuffle=False)
    model_ensemble.plot_fn(
        _confusion_matrix,
        test_data["dataset"],
        figure_path,
        labels=train_data["cat_labels"],
    )


def main():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    cli()


if __name__ == "__main__":
    main()
