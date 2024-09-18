import datetime
import os

import click
import pandas as pd
import tensorflow as tf
from biom import load_table
from sklearn.model_selection import KFold, StratifiedKFold

from aam.callbacks import (
    ConfusionMatrx,
    SaveModel,
    _confusion_matrix,
    _mean_absolute_error,
)
from aam.cv_utils import CVModel, EnsembleModel
from aam.losses import ImbalancedCategoricalCrossEntropy, ImbalancedMSE
from aam.transfer_nuc_model import TransferLearnNucleotideModel
from aam.unifrac_model import UnifracModel


@click.group()
class cli:
    pass


TABLE_DESC = (
    "Feature table containing all features that should be used for target prediction."
)
TEST_SIZE_DESC = "Fraction of input samples to exclude from training set and use for classifier testing."
CV_DESC = "Number of k-fold cross-validations to perform."
STRAT_DESC = "Evenly stratify training and test data among metadata categories. If True, all values in column must match at least two samples."
MISSING_SAMP_DESC = 'How to handle missing samples in metadata. "error" will fail if missing samples are detected. "ignore" will cause the feature table and metadata to be filtered, so that only samples found in both files are retained.'


@cli.command()
@click.option("--i-table", required=True, type=click.Path(exists=True), help=TABLE_DESC)
@click.option("--i-tree", required=True, type=click.Path(exists=True))
@click.option("--p-max-bp", required=True, type=int)
@click.option("--p-batch-size", default=8, type=int, show_default=True)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-ff-d-model", default=128, show_default=True, type=int)
@click.option("--p-pca-heads", default=8, show_default=True, type=int)
@click.option("--p-enc-layers", default=2, show_default=True, type=int)
@click.option("--p-enc-heads", default=8, show_default=True, type=int)
@click.option("--output-dir", required=True)
def fit_unifrac_regressor(
    i_table: str,
    i_tree: str,
    p_max_bp: int,
    p_batch_size: int,
    p_epochs: int,
    p_dropout: float,
    p_ff_d_model: int,
    p_pca_heads: int,
    p_enc_layers: int,
    p_enc_heads: int,
    output_dir: str,
):
    from aam.unifrac_data_utils import load_data

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    figure_path = os.path.join(output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    data_obj = load_data(i_table, tree_path=i_tree, batch_size=p_batch_size)

    load_model = False
    if load_model:
        model = tf.keras.models.load_model(f"{output_dir}/model.keras")
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
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(output_dir, log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_save_path = os.path.join(output_dir, "model.keras")
    model_saver = SaveModel(model_save_path, 1)
    core_callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
        ),
        tf.keras.callbacks.EarlyStopping("val_loss", patience=5, start_from_epoch=5),
        model_saver,
    ]
    model.fit(
        data_obj["training_dataset"],
        validation_data=data_obj["validation_dataset"],
        callbacks=[*core_callbacks],
        epochs=p_epochs,
    )
    model.set_weights(model_saver.best_weights)
    model.save(model_save_path, save_format="keras")


@cli.command()
@click.option(
    "--i-table",
    required=True,
    help=TABLE_DESC,
    type=click.Path(exists=True),
)
@click.option("--i-base-model-path", required=True, type=click.Path(exists=True))
@click.option(
    "--m-metadata-file",
    required=True,
    help="Metadata description",
    type=click.Path(exists=True),
)
@click.option(
    "--m-metadata-column",
    required=True,
    type=str,
    help="Numeric metadata column to use as prediction target.",
)
@click.option(
    "--p-missing-samples",
    default="error",
    type=click.Choice(["error", "ignore"], case_sensitive=False),
    help=MISSING_SAMP_DESC,
)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-mask-percent", default=25, show_default=True, type=int)
@click.option("--p-penalty", default=1, type=int)
@click.option("--p-cv", default=5, type=int, help=CV_DESC)
@click.option(
    "--p-test-size",
    default=0.2,
    show_default=True,
    type=click.FloatRange(0, 1),
    help=TEST_SIZE_DESC,
)
@click.option("--p-patience", default=10, show_default=True, type=int)
@click.option("--p-early-stop-warmup", default=50, show_default=True, type=int)
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
    p_patience: int,
    p_early_stop_warmup: int,
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
            patience=p_patience,
            early_stop_warmup=p_early_stop_warmup,
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
@click.option("--i-table", required=True, type=click.Path(exists=True), help=TABLE_DESC)
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
    help=MISSING_SAMP_DESC,
)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-mask-percent", default=25, show_default=True, type=int)
@click.option("--p-penalty", default=1, type=int)
@click.option("--p-cv", default=5, type=int, help=CV_DESC)
@click.option(
    "--p-test-size",
    default=0.2,
    show_default=True,
    type=click.FloatRange(0, 1),
    help=TEST_SIZE_DESC,
)
@click.option(
    "--p-stratify / --p-no-stratify", default=False, show_default=True, help=STRAT_DESC
)
@click.option("--p-patience", default=10, show_default=True, type=int)
@click.option("--p-early-stop-warmup", default=50, show_default=True, type=int)
@click.option("--p-report-back", default=1, show_default=True, type=int)
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
    p_stratify: bool,
    p_patience: int,
    p_early_stop_warmup: int,
    p_report_back: int,
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
    print(df)
    ids, table, df = validate_metadata(table, df, p_missing_samples)
    table, df = shuffle(table, df)
    num_ids = len(ids)
    categories = df[m_metadata_column].astype("category").cat.categories
    print("int", categories)
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
    if p_stratify:
        kfolds = StratifiedKFold(p_cv)
    else:
        kfolds = KFold(p_cv)

    cv_sample_ids = ids[fold_indices]
    sample_classes = df[df.index.isin(cv_sample_ids)][m_metadata_column]

    for i, (train_ind, val_ind) in enumerate(
        kfolds.split(fold_indices, sample_classes)
    ):
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
        loss = ImbalancedCategoricalCrossEntropy(train_data["cat_counts"])
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
            patience=p_patience,
            early_stop_warmup=p_early_stop_warmup,
            callbacks=[
                ConfusionMatrx(
                    dataset=val_data["dataset"],
                    output_dir=os.path.join(
                        figure_path, f"model-f{fold_label}-val.png"
                    ),
                    report_back=p_report_back,
                    labels=categories,
                )
            ],
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
