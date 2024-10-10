import datetime
import os

import click
import pandas as pd
import tensorflow as tf
from biom import load_table
from sklearn.model_selection import KFold, StratifiedKFold

from aam.callbacks import (
    ConfusionMatrx,
    MeanAbsoluteError,
    SaveModel,
    _confusion_matrix,
    _mean_absolute_error,
)
from aam.cv_utils import CVModel, EnsembleModel
from aam.losses import ImbalancedCategoricalCrossEntropy
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
@click.option("--p-enc-layers", default=4, show_default=True, type=int)
@click.option("--p-enc-heads", default=4, show_default=True, type=int)
@click.option("--p-penalty", default=0.01, type=float)
@click.option("--p-patience", default=10, show_default=True, type=int)
@click.option("--p-early-stop-warmup", default=50, show_default=True, type=int)
@click.option("--i-model", default="", required=False, type=str)
@click.option("--p-nuc-attention-heads", default=2, type=int)
@click.option("--p-nuc-attention-layers", default=4, type=int)
@click.option("--p-intermediate-ff", default=1024, type=int)
@click.option("--p-asv-limit", default=512, show_default=True, type=int)
@click.option(
    "--p-mixed-precision / --p-no-mixed-precision", default=True, required=False
)
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
    p_penalty: float,
    p_patience: int,
    p_early_stop_warmup: int,
    i_model: str,
    p_nuc_attention_heads: int,
    p_nuc_attention_layers: int,
    p_intermediate_ff: int,
    p_asv_limit: int,
    p_mixed_precision: bool,
    output_dir: str,
):
    from aam.unifrac_data_utils import load_data

    if p_mixed_precision:
        print("\nUsing mixed precision\n")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    figure_path = os.path.join(output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    if os.path.exists(i_model):
        model = tf.keras.models.load_model(i_model)
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
            penalty=p_penalty,
            nuc_attention_heads=p_nuc_attention_heads,
            nuc_attention_layers=p_nuc_attention_layers,
            intermediate_ff=p_intermediate_ff,
        )
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            3.2e-4,
            100000,
            end_learning_rate=1.28e-5,
            power=1.0,
            cycle=False,
        )

        optimizer = tf.keras.optimizers.AdamW(lr, beta_2=0.95)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.build()
    model.compile(
        optimizer=optimizer,
        run_eagerly=False,
    )
    data_obj = load_data(
        i_table,
        tree_path=i_tree,
        batch_size=p_batch_size,
        max_token_per_sample=p_asv_limit,
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
        tf.keras.callbacks.EarlyStopping(
            "val_loss", patience=p_patience, start_from_epoch=p_early_stop_warmup
        ),
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
    "--p-freeze-base-weights / --p-no-freeze-base-weights",
    default=True,
    required=False,
)
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
@click.option("--p-penalty", default=1, type=float)
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
@click.option("--p-batch-size", default=8, show_default=True, required=False, type=int)
@click.option("--p-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-report-back", default=5, show_default=True, type=int)
@click.option("--p-asv-limit", default=512, show_default=True, type=int)
@click.option(
    "--p-mixed-precision / --p-no-mixed-precision", default=True, required=False
)
@click.option("--output-dir", required=True, type=click.Path(exists=False))
def fit_sample_regressor(
    i_table: str,
    i_base_model_path: str,
    p_freeze_base_weights: bool,
    m_metadata_file: str,
    m_metadata_column: str,
    p_missing_samples: str,
    p_epochs: int,
    p_mask_percent: int,
    p_penalty: float,
    p_cv: int,
    p_test_size: float,
    p_patience: int,
    p_early_stop_warmup: int,
    p_batch_size: int,
    p_dropout: float,
    p_report_back: int,
    p_asv_limit: int,
    p_mixed_precision: bool,
    output_dir: str,
):
    from aam.transfer_data_utils import (
        load_data,
        shuffle,
        validate_metadata,
    )

    if p_mixed_precision:
        print("\nUsing mixed precision\n")
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

    def _get_fold(indices, shuffle, shift=None, scale=None):
        fold_ids = ids[indices]
        table_fold = table.filter(fold_ids, axis="sample", inplace=False)
        df_fold = df[df.index.isin(fold_ids)]
        data = load_data(
            table_fold,
            False,
            df_fold,
            m_metadata_column,
            shuffle_samples=shuffle,
            batch_size=p_batch_size,
            max_token_per_sample=p_asv_limit,
            shift=shift,
            scale=scale,
        )
        return data

    kfolds = KFold(p_cv)

    models = []
    for i, (train_ind, val_ind) in enumerate(kfolds.split(fold_indices)):
        train_data = _get_fold(train_ind, shuffle=True)
        val_data = _get_fold(
            val_ind, shuffle=False, shift=train_data["shift"], scale=train_data["scale"]
        )
        with open(os.path.join(model_path, f"f{i}_val_ids.txt"), "w") as f:
            for id in ids[val_ind]:
                f.write(id + "\n")

        base_model = tf.keras.models.load_model(i_base_model_path, compile=False)
        model = TransferLearnNucleotideModel(
            base_model,
            p_freeze_base_weights,
            mask_percent=p_mask_percent,
            shift=train_data["shift"],
            scale=train_data["scale"],
            penalty=p_penalty,
            dropout=p_dropout,
        )
        # loss = ImbalancedMSE(train_data["max_density"])
        loss = tf.keras.losses.Huber()
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
            metric="loss",
            patience=p_patience,
            early_stop_warmup=p_early_stop_warmup,
            callbacks=[
                MeanAbsoluteError(
                    dataset=val_data["dataset"],
                    output_dir=os.path.join(
                        figure_path, f"model_f{fold_label}-val.png"
                    ),
                    report_back=p_report_back,
                )
            ],
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
    help=TABLE_DESC,
    type=click.Path(exists=True),
)
@click.option("--i-model-path", required=True, type=click.Path(exists=True))
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
@click.option("--p-asv-limit", default=512, show_default=True, type=int)
@click.option("--p-batch-size", default=8, show_default=True, required=False, type=int)
@click.option(
    "--p-mixed-precision / --p-no-mixed-precision", default=True, required=False
)
@click.option("--output-dir", required=True, type=click.Path(exists=False))
def predict_sample_regressor(
    i_table: str,
    i_model_path: str,
    m_metadata_file: str,
    m_metadata_column: str,
    p_missing_samples: str,
    p_asv_limit: int,
    p_batch_size: int,
    p_mixed_precision: bool,
    output_dir: str,
):
    from aam.transfer_data_utils import (
        load_data,
        shuffle,
        validate_metadata,
    )

    if p_mixed_precision:
        print("\nUsing mixed precision\n")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    table = load_table(i_table)
    df = pd.read_csv(m_metadata_file, sep="\t", index_col=0)[[m_metadata_column]]
    ids, table, df = validate_metadata(table, df, p_missing_samples)
    table, df = shuffle(table, df)

    data = load_data(
        table,
        False,
        df,
        m_metadata_column,
        shuffle_samples=shuffle,
        batch_size=p_batch_size,
        max_token_per_sample=p_asv_limit,
    )

    model = tf.keras.models.load_model(i_model_path)

    y_pred, y_true = model.predict(data["dataset"])
    _mean_absolute_error(y_pred, y_true, os.path.join(output_dir, "mae.png"))


@cli.command()
@click.option("--i-table", required=True, type=click.Path(exists=True), help=TABLE_DESC)
@click.option("--i-base-model-path", required=True, type=click.Path(exists=True))
@click.option(
    "--p-freeze-base-weights / --p-no-freeze-base-weights",
    default=True,
    required=False,
)
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
@click.option("--p-penalty", default=1, type=float)
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
@click.option("--p-batch-size", default=8, show_default=True, required=False, type=int)
@click.option("--p-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-report-back", default=5, show_default=True, type=int)
@click.option("--p-asv-limit", default=512, show_default=True, type=int)
@click.option(
    "--p-mixed-precision / --p-no-mixed-precision", default=True, required=False
)
@click.option("--output-dir", required=True, type=click.Path(exists=False))
def fit_sample_classifier(
    i_table: str,
    i_base_model_path: str,
    p_freeze_base_weights: bool,
    m_metadata_file: str,
    m_metadata_column: str,
    p_missing_samples: str,
    p_epochs: int,
    p_mask_percent: int,
    p_penalty: float,
    p_cv: int,
    p_test_size: float,
    p_stratify: bool,
    p_patience: int,
    p_early_stop_warmup: int,
    p_batch_size: int,
    p_dropout: float,
    p_report_back: int,
    p_asv_limit: int,
    p_mixed_precision: bool,
    output_dir: str,
):
    from aam.transfer_data_utils import (
        load_data,
        shuffle,
        validate_metadata,
    )

    if p_mixed_precision:
        print("\nUsing mixed precision\n")
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
            table_fold,
            True,
            df_fold,
            m_metadata_column,
            shuffle_samples=shuffle,
            batch_size=p_batch_size,
            max_token_per_sample=p_asv_limit,
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
            p_freeze_base_weights,
            mask_percent=p_mask_percent,
            shift=train_data["shift"],
            scale=train_data["scale"],
            penalty=p_penalty,
            num_classes=train_data["num_classes"],
            dropout=p_dropout,
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
