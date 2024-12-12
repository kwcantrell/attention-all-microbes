from __future__ import annotations

import datetime
import os
from typing import Union

import click
import numpy as np
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
from aam.losses import ImbalancedCategoricalCrossEntropy


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


def validate_metadata(table, metadata, missing_samples_flag):
    # check for mismatch samples
    ids = table.ids(axis="sample")
    shared_ids = np.intersect1d(ids, metadata.index)
    min_ids = min(len(shared_ids), len(ids), len(metadata.index))
    max_ids = max(len(shared_ids), len(ids), len(metadata.index))
    if len(shared_ids) == 0:
        raise Exception("Table and Metadata have no matching sample ids")
    if min_ids != max_ids and missing_samples_flag == "error":
        raise Exception("Table and Metadata do not share all same sample ids.")
    elif min_ids != max_ids and missing_samples_flag == "ignore":
        print("Warning: Table and Metadata do not share all same sample ids.")
        print("Table and metadata will be filtered")
        table = table.filter(shared_ids, inplace=False)
        metadata = metadata.loc[table.ids()]
    return table.ids(), table, metadata


@cli.command()
@click.option("--i-table", required=True, type=click.Path(exists=True), help=TABLE_DESC)
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
@click.option("--p-batch-size", default=8, show_default=True, required=False, type=int)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-embedding-dim", default=128, type=int)
@click.option(
    "--p-intermediate-activation", default="relu", show_default=True, type=str
)
@click.option("--p-asv-limit", default=1024, show_default=True, type=int)
@click.option("--p-gen-new-table", default=True, show_default=True, type=bool)
@click.option("--p-lr", default=1e-4, show_default=True, type=float)
@click.option("--p-warmup-steps", default=10000, show_default=True, type=int)
@click.option("--p-decay-steps", default=1000, show_default=True, type=int)
@click.option("--p-max-bp", default=150, show_default=True, type=int)
@click.option("--output-dir", required=True)
@click.option("--p-add-token", default=False, required=False, type=bool)
@click.option("--p-is-categorical", default=False, required=False, type=bool)
@click.option("--p-rarefy-depth", default=5000, required=False, type=int)
@click.option("--p-weight-decay", default=0.004, show_default=True, type=float)
def fit_asv_encoder(
    i_table: str,
    m_metadata_file: str,
    m_metadata_column: str,
    p_missing_samples: bool,
    p_batch_size: int,
    p_epochs: int,
    p_dropout: float,
    p_embedding_dim: int,
    p_intermediate_activation: str,
    p_asv_limit: int,
    p_gen_new_table: bool,
    p_lr: float,
    p_warmup_steps: int,
    p_decay_steps: int,
    p_max_bp: int,
    output_dir: str,
    p_add_token: bool,
    p_is_categorical: bool,
    p_rarefy_depth: int,
    p_weight_decay: float,
):
    from biom import load_table

    from aam.data_handlers import GeneratorDataset
    from aam.models.nucleotide_encoder import NucleotideEncoder
    from aam.models.utils import cos_decay_with_warmup

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    figure_path = os.path.join(output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    model: tf.keras.Model = NucleotideEncoder(
        embedding_dim=p_embedding_dim,
        max_bp=p_max_bp,
        dropout_rate=p_dropout,
        intermediate_activation=p_intermediate_activation,
        add_token=p_add_token,
    )

    optimizer = tf.keras.optimizers.AdamW(
        cos_decay_with_warmup(p_lr, p_warmup_steps, p_decay_steps),
        weight_decay=p_weight_decay,
    )
    optimizer.exclude_from_weight_decay(
        var_names=[
            "bias",
            "rezero_alpha",
            "layer_norm",
            "LayerNorm",
            "embeddings",
        ]
    )
    token_shape = tf.TensorShape([None, None, 150])
    count_shape = tf.TensorShape([None, None, 1])
    model.build([token_shape, count_shape])
    model.compile(
        optimizer=optimizer,
        run_eagerly=False,
    )
    model.summary()

    table = load_table(i_table)
    df = pd.read_csv(m_metadata_file, sep="\t", index_col=0, dtype={0: str})[
        [m_metadata_column]
    ]
    ids, table, df = validate_metadata(table, df, p_missing_samples)
    indices = np.arange(len(ids), dtype=np.int32)

    np.random.shuffle(indices)
    train_size = int(len(ids) * 0.8)

    train_indices = indices[:train_size]
    train_ids = ids[train_indices]
    train_table = table.filter(train_ids, inplace=False)

    val_indices = indices[train_size:]
    val_ids = ids[val_indices]
    val_table = table.filter(val_ids, inplace=False)

    common_kwargs = {
        "metadata_column": m_metadata_column,
        "max_token_per_sample": p_asv_limit,
        "rarefy_depth": p_rarefy_depth,
        "batch_size": p_batch_size,
        "is_16S": True,
        "is_categorical": p_is_categorical,
        "max_bp": p_max_bp,
        "epochs": p_epochs,
        "metadata": df,
    }
    train_gen = GeneratorDataset(
        table=train_table,
        shuffle=True,
        shift=0.0,
        scale=1.0,
        gen_new_tables=p_gen_new_table,
        **common_kwargs,
    )
    train_data = train_gen.get_data()

    val_gen = GeneratorDataset(
        table=val_table,
        shuffle=False,
        shift=0.0,
        scale=1.0,
        gen_new_tables=False,
        **common_kwargs,
    )
    val_data = val_gen.get_data()

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(output_dir, log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_save_path = os.path.join(output_dir, "model.keras")
    model_saver = SaveModel(model_save_path, 1, monitor="val_loss")
    core_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        model_saver,
    ]
    model.fit(
        train_data["dataset"],
        validation_data=val_data["dataset"],
        callbacks=[*core_callbacks],
        epochs=p_epochs,
        steps_per_epoch=train_data["steps_pre_epoch"],
        validation_steps=val_data["steps_pre_epoch"],
    )
    model.set_weights(model_saver.best_weights)
    model.save(model_save_path, save_format="keras")


@cli.command()
@click.option("--i-table", required=True, type=click.Path(exists=True), help=TABLE_DESC)
@click.option("--i-tree", required=True, type=click.Path(exists=True))
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
@click.option("--p-batch-size", default=8, show_default=True, required=False, type=int)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-asv-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-patience", default=10, show_default=True, type=int)
@click.option("--p-early-stop-warmup", default=50, show_default=True, type=int)
@click.option("--i-model", default=None, required=False, type=str)
@click.option("--p-embedding-dim", default=128, type=int)
@click.option("--p-attention-heads", default=4, type=int)
@click.option("--p-attention-layers", default=4, type=int)
@click.option("--p-intermediate-size", default=1024, type=int)
@click.option(
    "--p-intermediate-activation", default="relu", show_default=True, type=str
)
@click.option("--p-asv-limit", default=1024, show_default=True, type=int)
@click.option("--p-gen-new-table", default=True, show_default=True, type=bool)
@click.option("--p-lr", default=1e-4, show_default=True, type=float)
@click.option("--p-warmup-steps", default=10000, show_default=True, type=int)
@click.option("--p-decay-steps", default=1000, show_default=True, type=int)
@click.option("--p-max-bp", default=150, show_default=True, type=int)
@click.option("--output-dir", required=True)
@click.option("--p-add-token", default=False, required=False, type=bool)
@click.option("--p-gotu", default=False, required=False, type=bool)
@click.option("--p-is-categorical", default=False, required=False, type=bool)
@click.option("--p-rarefy-depth", default=5000, required=False, type=int)
@click.option("--p-weight-decay", default=0.004, show_default=True, type=float)
@click.option("--p-accumulation-steps", default=1, required=False, type=int)
@click.option("--p-unifrac-metric", default="unifrac", required=False, type=str)
@click.option("--i-nucleotide-encoder", default=None, required=False, type=str)
@click.option("--p-loss-type", default="mse", required=False, type=str)
def fit_unifrac_regressor(
    i_table: str,
    i_tree: str,
    m_metadata_file: str,
    m_metadata_column: str,
    p_missing_samples: bool,
    p_batch_size: int,
    p_epochs: int,
    p_dropout: float,
    p_asv_dropout: float,
    p_patience: int,
    p_early_stop_warmup: int,
    i_model: Union[None, str],
    p_embedding_dim: int,
    p_attention_heads: int,
    p_attention_layers: int,
    p_intermediate_size: int,
    p_intermediate_activation: str,
    p_asv_limit: int,
    p_gen_new_table: bool,
    p_lr: float,
    p_warmup_steps: int,
    p_decay_steps: int,
    p_max_bp: int,
    output_dir: str,
    p_add_token: bool,
    p_gotu: bool,
    p_is_categorical: bool,
    p_rarefy_depth: int,
    p_weight_decay: float,
    p_accumulation_steps,
    p_unifrac_metric: str,
    i_nucleotide_encoder: str,
    p_loss_type: str,
):
    from biom import load_table

    from aam.data_handlers import UniFracGenerator
    from aam.models import SequenceEncoder
    from aam.models.utils import cos_decay_with_warmup

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    figure_path = os.path.join(output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    output_dim = p_embedding_dim
    if p_unifrac_metric == "faith_pd":
        output_dim = 1

    if i_model is not None:
        model = tf.keras.models.load_model(i_model, compile=False)
        model.accumulation_steps = p_accumulation_steps
    else:
        if i_nucleotide_encoder is not None:
            i_nucleotide_encoder = tf.keras.models.load_model(
                i_nucleotide_encoder, compile=False
            )
        model: tf.keras.Model = SequenceEncoder(
            output_dim,
            p_asv_limit,
            p_unifrac_metric,
            dropout_rate=p_dropout,
            embedding_dim=p_embedding_dim,
            attention_heads=p_attention_heads,
            attention_layers=p_attention_layers,
            intermediate_size=p_intermediate_size,
            intermediate_activation=p_intermediate_activation,
            max_bp=p_max_bp,
            is_16S=True,
            add_token=p_add_token,
            asv_dropout_rate=p_asv_dropout,
            accumulation_steps=p_accumulation_steps,
            nucleotide_encoder=i_nucleotide_encoder,
            pairwise_loss_type=p_loss_type,
        )

    optimizer = tf.keras.optimizers.AdamW(
        cos_decay_with_warmup(p_lr, p_warmup_steps, p_decay_steps),
        weight_decay=p_weight_decay,
    )
    optimizer.exclude_from_weight_decay(
        var_names=[
            "bias",
            "rezero_alpha",
            "layer_norm",
            "LayerNorm",
            "embeddings",
        ]
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    token_shape = tf.TensorShape([None, None, 150])
    count_shape = tf.TensorShape([None, None, 1])
    model.build([token_shape, count_shape])
    model.compile(
        optimizer=optimizer,
        run_eagerly=False,
    )
    model.summary()

    table = load_table(i_table)
    df = pd.read_csv(m_metadata_file, sep="\t", index_col=0, dtype={0: str})[
        [m_metadata_column]
    ]
    ids, table, df = validate_metadata(table, df, p_missing_samples)
    indices = np.arange(len(ids), dtype=np.int32)

    np.random.shuffle(indices)
    train_size = int(len(ids) * 0.8)

    train_indices = indices[:train_size]
    train_ids = ids[train_indices]
    train_table = table.filter(train_ids, inplace=False)

    val_indices = indices[train_size:]
    val_ids = ids[val_indices]
    val_table = table.filter(val_ids, inplace=False)

    common_kwargs = {
        "metadata_column": m_metadata_column,
        "max_token_per_sample": p_asv_limit,
        "rarefy_depth": p_rarefy_depth,
        "batch_size": p_batch_size,
        "is_16S": True,
        "is_categorical": p_is_categorical,
        "max_bp": p_max_bp,
        "epochs": p_epochs,
        "tree_path": i_tree,
        "metadata": df,
        "unifrac_metric": p_unifrac_metric,
    }
    train_gen = UniFracGenerator(
        table=train_table,
        shuffle=True,
        shift=0.0,
        scale=1.0,
        gen_new_tables=p_gen_new_table,
        **common_kwargs,
    )
    train_data = train_gen.get_data()

    val_gen = UniFracGenerator(
        table=val_table,
        shuffle=False,
        shift=0.0,
        scale=1.0,
        gen_new_tables=False,
        **common_kwargs,
    )
    val_data = val_gen.get_data()

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(output_dir, log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_save_path = os.path.join(output_dir, "model.keras")
    model_saver = SaveModel(model_save_path, 1, monitor="val_encoder_loss")
    core_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        # tf.keras.callbacks.EarlyStopping(
        #     "val_encoder_loss",
        #     patience=p_patience,
        #     start_from_epoch=p_early_stop_warmup,
        # ),
        model_saver,
    ]
    model.fit(
        train_data["dataset"],
        validation_data=val_data["dataset"],
        callbacks=[*core_callbacks],
        epochs=p_epochs,
        steps_per_epoch=train_data["steps_pre_epoch"],
        validation_steps=val_data["steps_pre_epoch"],
    )
    model.set_weights(model_saver.best_weights)
    model.save(model_save_path, save_format="keras")


@cli.command()
@click.option("--i-table", required=True, type=click.Path(exists=True), help=TABLE_DESC)
@click.option("--i-taxonomy", required=True, type=click.Path(exists=True))
@click.option("--i-tax-level", default=7, type=int)
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
@click.option("--p-batch-size", default=8, show_default=True, required=False, type=int)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-dropout", default=0.1, show_default=True, type=float)
@click.option("--p-asv-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-patience", default=10, show_default=True, type=int)
@click.option("--p-early-stop-warmup", default=50, show_default=True, type=int)
@click.option("--i-model", default=None, required=False, type=str)
@click.option("--p-embedding-dim", default=128, type=int)
@click.option("--p-attention-heads", default=4, type=int)
@click.option("--p-attention-layers", default=4, type=int)
@click.option("--p-intermediate-size", default=1024, type=int)
@click.option(
    "--p-intermediate-activation", default="relu", show_default=True, type=str
)
@click.option("--p-asv-limit", default=512, show_default=True, type=int)
@click.option("--p-gen-new-table", default=True, show_default=True, type=bool)
@click.option("--p-lr", default=1e-4, show_default=True, type=float)
@click.option("--p-warmup-steps", default=10000, show_default=True, type=int)
@click.option("--p-decay-steps", default=1000, show_default=True, type=int)
@click.option("--p-max-bp", required=True, type=int)
@click.option("--output-dir", required=True)
@click.option("--p-add-token", default=False, required=False, type=bool)
@click.option("--p-gotu", default=False, required=False, type=bool)
@click.option("--p-is-categorical", default=False, required=False, type=bool)
@click.option("--p-rarefy-depth", default=5000, required=False, type=int)
@click.option("--p-weight-decay", default=0.004, show_default=True, type=float)
@click.option("--p-accumulation-steps", default=1, required=False, type=int)
def fit_taxonomy_regressor(
    i_table: str,
    i_taxonomy: str,
    i_tax_level: int,
    m_metadata_file: str,
    m_metadata_column: str,
    p_missing_samples: bool,
    p_batch_size: int,
    p_epochs: int,
    p_dropout: float,
    p_asv_dropout: float,
    p_patience: int,
    p_early_stop_warmup: int,
    i_model: Union[None, str],
    p_embedding_dim: int,
    p_attention_heads: int,
    p_attention_layers: int,
    p_intermediate_size: int,
    p_intermediate_activation: str,
    p_asv_limit: int,
    p_gen_new_table: bool,
    p_lr: float,
    p_warmup_steps: int,
    p_decay_steps: int,
    p_max_bp: int,
    output_dir: str,
    p_add_token: bool,
    p_gotu: bool,
    p_is_categorical: bool,
    p_rarefy_depth: int,
    p_weight_decay: float,
    p_accumulation_steps,
):
    from biom import load_table

    from aam.data_handlers import TaxonomyGenerator
    from aam.models import SequenceEncoder
    from aam.models.utils import cos_decay_with_warmup

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    figure_path = os.path.join(output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    table = load_table(i_table)
    df = pd.read_csv(m_metadata_file, sep="\t", index_col=0, dtype={0: str})[
        [m_metadata_column]
    ]
    ids, table, df = validate_metadata(table, df, p_missing_samples)
    indices = np.arange(len(ids), dtype=np.int32)

    np.random.shuffle(indices)
    train_size = int(len(ids) * 0.8)

    train_indices = indices[:train_size]
    train_ids = ids[train_indices]
    train_table = table.filter(train_ids, inplace=False)

    val_indices = indices[train_size:]
    val_ids = ids[val_indices]
    val_table = table.filter(val_ids, inplace=False)

    common_kwargs = {
        "metadata_column": m_metadata_column,
        "max_token_per_sample": p_asv_limit,
        "rarefy_depth": p_rarefy_depth,
        "batch_size": p_batch_size,
        "is_16S": True,
        "is_categorical": p_is_categorical,
        "max_bp": p_max_bp,
        "epochs": p_epochs,
        "taxonomy": i_taxonomy,
        "tax_level": i_tax_level,
        "metadata": df,
    }
    train_gen = TaxonomyGenerator(
        table=train_table,
        shuffle=True,
        shift=0.0,
        scale=1.0,
        gen_new_tables=p_gen_new_table,
        **common_kwargs,
    )
    train_data = train_gen.get_data()

    val_gen = TaxonomyGenerator(
        table=val_table,
        shuffle=False,
        shift=0.0,
        scale=1.0,
        gen_new_tables=False,
        **common_kwargs,
    )
    val_data = val_gen.get_data()

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(output_dir, log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_save_path = os.path.join(output_dir, "model.keras")
    model_saver = SaveModel(model_save_path, 1, monitor="val_encoder_loss")
    core_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        # tf.keras.callbacks.EarlyStopping(
        #     "val_loss", patience=p_patience, start_from_epoch=p_early_stop_warmup
        # ),
        model_saver,
    ]

    if i_model is not None:
        model: tf.keras.Model = tf.keras.models.load_model(i_model)
    else:
        model: tf.keras.Model = SequenceEncoder(
            train_gen.num_tokens,
            p_asv_limit,
            "taxonomy",
            dropout_rate=p_dropout,
            embedding_dim=p_embedding_dim,
            attention_heads=p_attention_heads,
            attention_layers=p_attention_layers,
            intermediate_size=p_intermediate_size,
            intermediate_activation=p_intermediate_activation,
            max_bp=p_max_bp,
            is_16S=True,
            add_token=p_add_token,
            asv_dropout_rate=p_asv_dropout,
            accumulation_steps=p_accumulation_steps,
        )
    optimizer = tf.keras.optimizers.AdamW(
        cos_decay_with_warmup(p_lr, p_warmup_steps, p_decay_steps),
        weight_decay=p_weight_decay,
    )
    optimizer.exclude_from_weight_decay(
        var_names=[
            "bias",
            "rezero_alpha",
            "layer_norm",
            "LayerNorm",
            "embeddings",
        ]
    )
    token_shape = tf.TensorShape([None, None, 150])
    count_shape = tf.TensorShape([None, None, 1])
    model.build([token_shape, count_shape])
    model.compile(
        optimizer=optimizer,
        run_eagerly=False,
    )
    model.summary()
    model.fit(
        train_data["dataset"],
        validation_data=val_data["dataset"],
        callbacks=[*core_callbacks],
        epochs=p_epochs,
        steps_per_epoch=train_data["steps_pre_epoch"],
        validation_steps=val_data["steps_pre_epoch"],
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
@click.option(
    "--i-base-model-path", default=None, required=False, type=click.Path(exists=True)
)
@click.option(
    "--p-no-freeze-base-weights / --p-freeze-base-weights",
    default=False,
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
@click.option("--p-dropout", default=0.1, show_default=True, type=float)
@click.option("--p-asv-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-report-back", default=5, show_default=True, type=int)
@click.option("--p-asv-limit", default=1024, show_default=True, type=int)
@click.option("--p-penalty", default=1.0, show_default=True, type=float)
@click.option("--p-nuc-penalty", default=1.0, show_default=True, type=float)
@click.option("--p-embedding-dim", default=128, show_default=True, type=int)
@click.option("--p-attention-heads", default=4, show_default=True, type=int)
@click.option("--p-attention-layers", default=4, show_default=True, type=int)
@click.option("--p-intermediate-size", default=1024, show_default=True, type=int)
@click.option(
    "--p-intermediate-activation", default="relu", show_default=True, type=str
)
@click.option("--p-taxonomy", default=None, type=click.Path(exists=True))
@click.option("--p-taxonomy-level", default=7, show_default=True, type=int)
@click.option("--p-tree", default=None, type=click.Path(exists=True))
@click.option("--p-gen-new-table", default=True, show_default=True, type=bool)
@click.option("--p-lr", default=1e-4, show_default=True, type=float)
@click.option("--p-warmup-steps", default=4000, show_default=True, type=int)
@click.option("--p-decay-steps", default=1000, show_default=True, type=int)
@click.option("--p-max-bp", default=150, show_default=True, type=int)
@click.option("--output-dir", required=True, type=click.Path(exists=False))
@click.option("--p-output-dim", default=1, required=False, type=int)
@click.option("--p-add-token", default=False, required=False, type=bool)
@click.option("--p-gotu", default=False, required=False, type=bool)
@click.option("--p-is-categorical", default=False, required=False, type=bool)
@click.option("--p-rarefy-depth", default=5000, required=False, type=int)
@click.option("--p-weight-decay", default=0.004, show_default=True, type=float)
@click.option("--p-accumulation-steps", default=1, required=False, type=int)
@click.option("--p-unifrac-metric", default="unifrac", required=False, type=str)
@click.option("--p-scale-loss", default=False, type=bool)
def fit_sample_regressor(
    i_table: str,
    i_base_model_path: str,
    p_no_freeze_base_weights: bool,
    m_metadata_file: str,
    m_metadata_column: str,
    p_missing_samples: str,
    p_epochs: int,
    p_cv: int,
    p_test_size: float,
    p_patience: int,
    p_early_stop_warmup: int,
    p_batch_size: int,
    p_dropout: float,
    p_asv_dropout: float,
    p_report_back: int,
    p_asv_limit: int,
    p_penalty: float,
    p_nuc_penalty: float,
    p_embedding_dim: int,
    p_attention_heads: int,
    p_attention_layers: int,
    p_intermediate_size: int,
    p_intermediate_activation: str,
    p_taxonomy: str,
    p_taxonomy_level: int,
    p_tree: str,
    p_gen_new_table: bool,
    p_lr: int,
    p_warmup_steps: int,
    p_decay_steps: int,
    p_max_bp: int,
    output_dir: str,
    p_output_dim: int,
    p_add_token: bool,
    p_gotu: bool,
    p_is_categorical: bool,
    p_rarefy_depth: int,
    p_weight_decay: float,
    p_accumulation_steps: int,
    p_unifrac_metric: str,
    p_scale_loss: bool,
):
    from aam.callbacks import ConfusionMatrx
    from aam.data_handlers import CombinedGenerator, TaxonomyGenerator, UniFracGenerator
    from aam.models import SequenceRegressor

    # p_is_16S = False
    is_16S = not p_gotu
    # p_is_categorical = True
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    figure_path = os.path.join(output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    model_path = os.path.join(output_dir, "cv-models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    table = load_table(i_table)
    df = pd.read_csv(m_metadata_file, sep="\t", index_col=0, dtype={0: str})[
        [m_metadata_column]
    ]
    ids, table, df = validate_metadata(table, df, p_missing_samples)
    num_ids = len(ids)

    fold_indices = np.arange(num_ids)
    np.random.shuffle(fold_indices)
    if p_test_size > 0:
        test_size = int(num_ids * p_test_size)
        train_size = num_ids - test_size
        test_indices = fold_indices[train_size:]
        fold_indices = fold_indices[:train_size]

    print(len(test_indices), len(fold_indices))

    common_kwargs = {
        "metadata_column": m_metadata_column,
        "max_token_per_sample": p_asv_limit,
        "rarefy_depth": p_rarefy_depth,
        "batch_size": p_batch_size,
        "is_16S": is_16S,
        "is_categorical": p_is_categorical,
    }

    def tax_gen(table, df, shuffle, shift, scale, epochs, gen_new_tables):
        return TaxonomyGenerator(
            table=table,
            metadata=df,
            taxonomy=p_taxonomy,
            tax_level=p_taxonomy_level,
            shuffle=shuffle,
            shift=shift,
            scale=scale,
            epochs=epochs,
            gen_new_tables=gen_new_tables,
            max_bp=p_max_bp,
            **common_kwargs,
        )

    def unifrac_gen(table, df, shuffle, shift, scale, epochs, gen_new_tables):
        return UniFracGenerator(
            table=table,
            metadata=df,
            tree_path=p_tree,
            shuffle=shuffle,
            shift=shift,
            scale=scale,
            epochs=epochs,
            gen_new_tables=gen_new_tables,
            max_bp=p_max_bp,
            unifrac_metric=p_unifrac_metric,
            **common_kwargs,
        )

    def combine_gen(table, df, shuffle, shift, scale, epochs, gen_new_tables):
        return CombinedGenerator(
            table=table,
            metadata=df,
            tree_path=p_tree,
            taxonomy=p_taxonomy,
            tax_level=p_taxonomy_level,
            shuffle=shuffle,
            shift=shift,
            scale=scale,
            epochs=epochs,
            gen_new_tables=gen_new_tables,
            max_bp=p_max_bp,
            **common_kwargs,
        )

    if p_unifrac_metric == "combined":
        base_model = "combined"
        generator = combine_gen
    elif p_taxonomy is not None and p_tree is None:
        base_model = "taxonomy"
        generator = tax_gen
    elif p_taxonomy is None and p_tree is not None:
        base_model = p_unifrac_metric
        generator = unifrac_gen
    else:
        raise Exception("Only taxonomy or UniFrac is supported.")

    if i_base_model_path is not None:
        base_model = tf.keras.models.load_model(i_base_model_path, compile=False)
        base_type = base_model.encoder_type
        if base_type == "taxonomy":
            generator = tax_gen
        else:
            generator = unifrac_gen

        if not p_no_freeze_base_weights:
            print("base_model's weights are set to trainable.")

    def _get_fold(
        indices,
        shuffle,
        shift=None,
        scale=None,
        epochs=1000,
        gen_new_tables=False,
    ):
        fold_ids = ids[indices]
        table_fold = table.filter(fold_ids, axis="sample", inplace=False)
        df_fold = df.loc[fold_ids]

        gen = generator(
            table_fold, df_fold, shuffle, shift, scale, epochs, gen_new_tables
        )

        data = gen.get_data()
        if hasattr(gen, "num_tokens"):
            data["num_tokens"] = gen.num_tokens
        else:
            data["num_tokens"] = None
        return data

    if not p_is_categorical:
        print("non-stratified folds")
        kfolds = KFold(p_cv)
        splits = kfolds.split(fold_indices)
    else:
        print("stratified folds...")
        kfolds = StratifiedKFold(p_cv)
        train_ids = ids[fold_indices]
        train_classes = df.loc[df.index.isin(train_ids), m_metadata_column].values
        splits = kfolds.split(fold_indices, train_classes)

    models = []
    for i, (train_ind, val_ind) in enumerate(splits):
        train_data = _get_fold(
            train_ind,
            shuffle=True,
            # shift=0.0,
            scale="standscale",
            gen_new_tables=p_gen_new_table,
        )
        val_data = _get_fold(
            val_ind,
            shuffle=False,
            shift=train_data["shift"],
            scale=train_data["scale"],
            epochs=1,
        )
        with open(os.path.join(model_path, f"f{i}_val_ids.txt"), "w") as f:
            for id in ids[val_ind]:
                f.write(id + "\n")
        vocab_size = 6 if not p_is_categorical else 2000

        if base_model == "combined":
            base_output_dim = [p_embedding_dim, 1, train_data["num_tokens"]]
        elif base_model == "unifrac":
            base_output_dim = p_embedding_dim
        elif base_model == "faith_pd":
            base_output_dim = 1
        else:
            base_output_dim = train_data["num_tokens"]

        model = SequenceRegressor(
            token_limit=p_asv_limit,
            base_output_dim=base_output_dim,
            shift=train_data["shift"],
            scale=train_data["scale"],
            dropout_rate=p_dropout,
            embedding_dim=p_embedding_dim,
            attention_heads=p_attention_heads,
            attention_layers=p_attention_layers,
            intermediate_size=p_intermediate_size,
            intermediate_activation=p_intermediate_activation,
            base_model=base_model,
            freeze_base=p_no_freeze_base_weights,
            penalty=p_penalty,
            nuc_penalty=p_nuc_penalty,
            max_bp=p_max_bp,
            is_16S=is_16S,
            vocab_size=vocab_size,
            out_dim=p_output_dim,
            classifier=p_is_categorical,
            add_token=p_add_token,
            class_weights=train_data["class_weights"],
            accumulation_steps=p_accumulation_steps,
            scale_losses=p_scale_loss,
        )
        token_shape = tf.TensorShape([None, None, p_max_bp])
        count_shape = tf.TensorShape([None, None, 1])
        model.build([token_shape, count_shape])
        model.summary()

        fold_label = i + 1
        if not p_is_categorical:
            loss = tf.keras.losses.MeanSquaredError(reduction="none")
            callbacks = [
                # MeanAbsoluteError(
                #     monitor="val_mae",
                #     dataset=val_data["dataset"],
                #     output_dir=os.path.join(
                #         figure_path, f"model_f{fold_label}-val.png"
                #     ),
                #     report_back=p_report_back,
                # )
            ]
        else:
            loss = tf.keras.losses.CategoricalFocalCrossentropy(
                from_logits=False, reduction="none"
            )
            # loss = tf.keras.losses.CategoricalHinge(reduction="none")
            callbacks = [
                ConfusionMatrx(
                    monitor="val_target_loss",
                    dataset=val_data["dataset"],
                    output_dir=os.path.join(
                        figure_path, f"model_f{fold_label}-val.png"
                    ),
                    report_back=p_report_back,
                )
            ]
        model_cv = CVModel(
            model,
            train_data,
            val_data,
            output_dir,
            fold_label,
        )
        metric = "mae" if not p_is_categorical else "target_loss"
        model_cv.fit_fold(
            loss,
            p_epochs,
            os.path.join(model_path, f"model_f{fold_label}.keras"),
            metric=metric,
            patience=p_patience,
            early_stop_warmup=p_early_stop_warmup,
            callbacks=[*callbacks],
            lr=p_lr,
            warmup_steps=p_warmup_steps,
            decay_steps=p_decay_steps,
            weight_decay=p_weight_decay,
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

    test_data = _get_fold(
        test_indices,
        shuffle=False,
        shift=train_data["shift"],
        scale=train_data["scale"],
        epochs=1,
        num_tables=5,
    )
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
    from aam.transfer_nuc_model import TransferLearnNucleotideModel

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
    fold_indices = list(range(num_ids))
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
            train_data,
            val_data,
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




@cli.command()
@click.option(
    "--i-asv-table",
    required=True,
    help=TABLE_DESC,
    type=click.Path(exists=True),
)
@click.option(
    "--i-gotu-table",
    required=True,
    help=TABLE_DESC,
    type=click.Path(exists=True),
)
@click.option(
    "--i-base-model-path", default=None, required=False, type=click.Path(exists=True)
)
@click.option(
    "--p-no-freeze-base-weights / --p-freeze-base-weights",
    default=False,
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
@click.option("--p-dropout", default=0.1, show_default=True, type=float)
@click.option("--p-asv-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-report-back", default=5, show_default=True, type=int)
@click.option("--p-asv-limit", default=1024, show_default=True, type=int)
@click.option("--p-penalty", default=1.0, show_default=True, type=float)
@click.option("--p-nuc-penalty", default=1.0, show_default=True, type=float)
@click.option("--p-embedding-dim", default=128, show_default=True, type=int)
@click.option("--p-attention-heads", default=4, show_default=True, type=int)
@click.option("--p-attention-layers", default=4, show_default=True, type=int)
@click.option("--p-intermediate-size", default=1024, show_default=True, type=int)
@click.option(
    "--p-intermediate-activation", default="relu", show_default=True, type=str
)
@click.option("--p-taxonomy", default=None, type=click.Path(exists=True))
@click.option("--p-taxonomy-level", default=7, show_default=True, type=int)
@click.option("--p-tree", default=None, type=click.Path(exists=True))
@click.option("--p-gen-new-table", default=True, show_default=True, type=bool)
@click.option("--p-lr", default=1e-4, show_default=True, type=float)
@click.option("--p-warmup-steps", default=4000, show_default=True, type=int)
@click.option("--p-decay-steps", default=1000, show_default=True, type=int)
@click.option("--p-max-bp", default=150, show_default=True, type=int)
@click.option("--output-dir", required=True, type=click.Path(exists=False))
@click.option("--p-output-dim", default=128, required=False, type=int)
@click.option("--p-add-token", default=False, required=False, type=bool)
@click.option("--p-is-categorical", default=False, required=False, type=bool)
@click.option("--p-rarefy-depth", default=5000, required=False, type=int)
@click.option("--p-weight-decay", default=0.004, show_default=True, type=float)
@click.option("--p-accumulation-steps", default=1, required=False, type=int)
@click.option("--p-unifrac-metric", default="unifrac", required=False, type=str)
@click.option("--p-scale-loss", default=False, type=bool)
def fit_gotu(
    i_asv_table: str,
    i_gotu_table: str,
    i_base_model_path: str,
    p_no_freeze_base_weights: bool,
    m_metadata_file: str,
    m_metadata_column: str,
    p_missing_samples: str,
    p_epochs: int,
    p_cv: int,
    p_test_size: float,
    p_patience: int,
    p_early_stop_warmup: int,
    p_batch_size: int,
    p_dropout: float,
    p_asv_dropout: float,
    p_report_back: int,
    p_asv_limit: int,
    p_penalty: float,
    p_nuc_penalty: float,
    p_embedding_dim: int,
    p_attention_heads: int,
    p_attention_layers: int,
    p_intermediate_size: int,
    p_intermediate_activation: str,
    p_taxonomy: str,
    p_taxonomy_level: int,
    p_tree: str,
    p_gen_new_table: bool,
    p_lr: int,
    p_warmup_steps: int,
    p_decay_steps: int,
    p_max_bp: int,
    output_dir: str,
    p_output_dim: int,
    p_add_token: bool,
    p_is_categorical: bool,
    p_rarefy_depth: int,
    p_weight_decay: float,
    p_accumulation_steps: int,
    p_unifrac_metric: str,
    p_scale_loss: bool,
):
    from aam.data_handlers import GOTUGenerator
    from aam.models import GOTUModel
    from aam.models.utils import cos_decay_with_warmup
    
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    figure_path = os.path.join(output_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    
    asv_table = load_table(i_asv_table)
    gotu_table = load_table(i_gotu_table)

    df_all = pd.read_csv(m_metadata_file, sep="\t", index_col=0, dtype={0: str})[
        [m_metadata_column]
    ]
    asv_ids, asv_table, df = validate_metadata(asv_table, df_all, p_missing_samples)
    gotu_ids, gotu_table, df = validate_metadata(gotu_table, df_all, p_missing_samples)
    num_ids = len(gotu_ids)
    gotu_count = len(gotu_table.ids(axis='observation'))


    common_kwargs = {
        "metadata_column": m_metadata_column,
        "max_token_per_sample": p_asv_limit,
        "rarefy_depth": p_rarefy_depth,
        "batch_size": p_batch_size,
        "is_16S": True,
        "is_categorical": p_is_categorical,
    }

    def train_generator(
        asv_table, gotu_table, df, shuffle, shift, scale, epochs, gen_new_tables
    ):
        return GOTUGenerator(
            table=gotu_table,
            asv_table=asv_table,
            shuffle=shuffle,
            shift=shift,
            scale=scale,
            gen_new_tables=gen_new_tables,
            **common_kwargs,
        )
    def val_generator(
        asv_table, gotu_table, df, shuffle, shift, scale, epochs, gen_new_tables
    ):
        return GOTUGenerator(
            table=gotu_table,
            asv_table=asv_table,
            shuffle=shuffle,
            shift=shift,
            scale=scale,
            gen_new_tables=gen_new_tables,
            **common_kwargs,
        )

    
    indices = np.arange(len(asv_ids), dtype=np.int32)

    np.random.shuffle(indices)
    train_size = int(len(asv_ids) * 0.8)

    train_indices = indices[:train_size]
    train_asv_ids = asv_ids[train_indices]
    train_gotu_ids = gotu_ids[train_indices]
    train_asv_table = asv_table.filter(train_asv_ids, inplace=False)
    train_gotu_table = gotu_table.filter(train_gotu_ids, inplace=False)

    val_asv_indices = indices[train_size:]
    val_asv_ids = asv_ids[val_asv_indices]
    val_asv_table = asv_table.filter(val_asv_ids, inplace=False)
    
    val_gotu_indices = indices[train_size:]
    val_gotu_ids = gotu_ids[val_gotu_indices]
    val_gotu_table = gotu_table.filter(val_gotu_ids, inplace=False)
    
    common_kwargs = {
        "metadata_column": m_metadata_column,
        "max_token_per_sample": p_asv_limit,
        "rarefy_depth": p_rarefy_depth,
        "batch_size": p_batch_size,
        "is_16S": True,
        "is_categorical": p_is_categorical,
        "max_bp": p_max_bp,
        "epochs": p_epochs,
        "tree_path": p_tree,
        "metadata": df_all,
    }
    
    train_gen = train_generator(
        train_asv_table, train_gotu_table, df_all, True, 0, 1, p_epochs, True
    )
    
    val_gen = val_generator(
        val_asv_table, val_gotu_table, df_all, False, 0, 1, p_epochs, False
    )
    
    train_data = train_gen.get_data()
    val_data = val_gen.get_data()
    base_model = None
    if i_base_model_path is not None:
        base_model = tf.keras.models.load_model(i_base_model_path, compile=False)
        base_model.accumulation_steps = p_accumulation_steps
    model = GOTUModel(
        p_output_dim,
        p_asv_limit,
        dropout_rate=p_dropout,
        embedding_dim=p_embedding_dim,
        attention_heads=p_attention_heads,
        attention_layers=p_attention_layers,
        intermediate_size=p_intermediate_size,
        intermediate_activation=p_intermediate_activation,
        base_model=base_model,
        gotu_count=gotu_count,
        freeze_base_weights=p_no_freeze_base_weights,
    )
    asv_shape = tf.TensorShape(([None, None, p_max_bp], [None, None, 1]))
    gotu_shape = tf.TensorShape(([None, None, 1], [None, None, 1]))
    optimizer = tf.keras.optimizers.AdamW(
        cos_decay_with_warmup(p_lr, p_warmup_steps, p_decay_steps),
        weight_decay=p_weight_decay,
    )
    optimizer.exclude_from_weight_decay(
        var_names=[
            "bias",
            "rezero_alpha",
            "layer_norm",
            "LayerNorm",
            "embeddings",
        ]
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    
    model.build((asv_shape, gotu_shape))
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
    model_saver = SaveModel(model_save_path, 1, monitor="val_loss")
    core_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        # tf.keras.callbacks.EarlyStopping(
        #     "val_encoder_loss",
        #     patience=p_patience,
        #     start_from_epoch=p_early_stop_warmup,
        # ),
        model_saver,
    ]
    model.fit(
        train_data["dataset"],
        validation_data=val_data["dataset"],
        callbacks=[*core_callbacks],
        epochs=p_epochs,
        steps_per_epoch=train_data["steps_pre_epoch"],
        validation_steps=val_data["steps_pre_epoch"],
    )
    model.set_weights(model_saver.best_weights)
    model.save(model_save_path, save_format="keras")

def main():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    cli()


if __name__ == "__main__":
    main()
