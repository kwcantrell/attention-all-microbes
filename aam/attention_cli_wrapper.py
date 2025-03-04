from __future__ import annotations

import datetime
import os
from typing import Union
from functools import wraps
import json
import os
import click
import numpy as np
import pandas as pd
import tensorflow as tf
from attention_wrappers import (
    fit_asv_encoder_decorator,
    fit_denoised_unifrac_regressor_decorator,
    fit_gotu_decorator,
    fit_sample_regressor_decorator,
    fit_taxonomy_regressor_decorator,
    predict_sample_regressor_decorator,
)

from cli_wrapper_tests import (
    validate_fit_asv_encoder,
    validate_fit_denoised_unifrac_regressor,
    validate_fit_taxonomy_regressor,
    validate_fit_sample_regressor,
    validate_predict_sample_regressor,
    validate_gotu_infer
)

from biom import load_table
from bp import parse_newick, to_skbio_treenode
from sklearn.model_selection import KFold, StratifiedKFold

from aam.callbacks import (
    ConfusionMatrx,
    SaveModel,
    _confusion_matrix,
    _mean_absolute_error,
)
from aam.cv_utils import CVModel, EnsembleModel
import pytest
from click.testing import CliRunner
class CommonParams:
    """Singleton for storing, retrieving, and managing common CLI parameters with persistence."""
    _instance = None
    _file_path = "common_params.json"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CommonParams, cls).__new__(cls)
            cls._instance._params = cls._load_params()
        return cls._instance

    @classmethod
    def _load_params(cls):
        """Load parameters from the file if it exists, otherwise return an empty dict."""
        if os.path.exists(cls._file_path):
            try:
                with open(cls._file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}  # Return empty dict if JSON is corrupted
        return {}

    def save_params(self, **kwargs):
        """Save provided parameters and persist them to a file."""
        self._params.update(kwargs)
        with open(self._file_path, "w") as f:
            json.dump(self._params, f, indent=4)
        click.echo("Common parameters saved!")

    def get_params(self):
        """Retrieve all stored common parameters."""
        return self._params

    def delete_params(self):
        """Delete all stored common parameters and remove the file."""
        self._params = {}
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        click.echo("Common parameters deleted!")

def inject_common_params(func):
    """Decorator to inject stored common parameters into CLI functions."""
    @wraps(func)
    def wrapper(**kwargs):
        with open ("common_params.json", "r") as f:
            saved_params = json.load(f)

        for key, value in saved_params.items():
            if key not in kwargs or kwargs[key] is None:
                kwargs[key] = value

        print("Final kwargs after merging:", kwargs)

        return func(**kwargs)
    return wrapper


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

GLOBAL_CONFIGURATIONS = {}

@cli.command()
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
@click.option("--p-patience", default=10, show_default=True, type=int)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-weight-decay", default=0.004, show_default=True, type=float)
@click.option("--p-gen-new-table", default=True, show_default=True, type=bool)
@click.option("--p-lr", default=1e-4, show_default=True, type=float)
@click.option("--p-attention-layers", default=4, type=int)
@click.option("--p-early-stop-warmup", default=50, show_default=True, type=int)
@click.option("--p-batch-size", default=8, show_default=True, required=False, type=int)
@click.option("--p-decay-steps", show_default=True, type=int)
@click.option("--p-embedding-dim", default=128, type=int)
@click.option("--p-asv-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-rarefy-depth", default=5000, required=False, type=int)
@click.option("--p-attention-heads", default=4, type=int)
@click.option("--i-table", show_default=True, type=str)
@click.option("--p-dropout", default=0.1, show_default=True, type=float)
@click.option(
    "--m-metadata-file",
    required=True,
    help="Metadata description",
    type=click.Path(exists=True),
)
@click.option(
    "--p-intermediate-activation", default="relu", show_default=True, type=str
)
@click.option("--p-asv-limit", default=512, show_default=True, type=int)
@click.option("--p-max-bp", type=int)
@click.option("--p-is-categorical", default=False, required=False, type=bool)
@click.option("--i-model", default=None, required=False, type=str)
@click.option("--output-dir", required = True)
@click.option("--p-gotu", default=False, required=False, type=bool)
@click.option("--p-add-token", default=False, required=False, type=bool)
@click.option("--p-warmup-steps", default=10000, show_default=True, type=int)
@click.option("--p-intermediate-size", default=1024, type=int)
def save_common_params_fitunitax(m_metadata_column,
                                     p_missing_samples,
                                     p_patience,
                                     p_epochs,
                                     p_weight_decay,
                                     p_gen_new_table,
                                     p_lr,
                                     p_attention_layers,
                                     p_early_stop_warmup,
                                     p_batch_size,
                                     p_decay_steps,
                                     p_embedding_dim,
                                     p_asv_dropout,
                                     p_rarefy_depth,
                                     p_attention_heads,
                                     i_table,
                                     p_dropout,
                                     m_metadata_file,
                                     p_intermediate_activation,
                                     p_asv_limit,
                                     p_max_bp,
                                     p_is_categorical,
                                     i_model,
                                     output_dir,
                                     p_gotu,
                                     p_add_token,
                                     p_warmup_steps,
                                     p_intermediate_size):
    """Save common parameters for reuse."""
    CommonParams().save_params(
        i_table=i_table,
        output_dir=output_dir,
        m_metadata_column=m_metadata_column,
        m_metadata_file=m_metadata_file,
        p_missing_samples= p_missing_samples,
        p_patience=p_patience,
        p_epochs=p_epochs,
        p_weight_decay=p_weight_decay,
        p_gen_new_table=p_gen_new_table,
        p_lr=p_lr,
        p_attention_layers=p_attention_layers,
        p_early_stop_warmup=p_early_stop_warmup,
        p_batch_size=p_batch_size,
        p_decay_steps=p_decay_steps,
        p_embedding_dim=p_embedding_dim,
        p_asv_dropout=p_asv_dropout,
        p_rarefy_depth=p_rarefy_depth,
        p_attention_heads=p_attention_heads,
        p_dropout=p_dropout,
        p_intermediate_activation=p_intermediate_activation,
        p_asv_limit=p_asv_limit,
        p_max_bp=p_max_bp,
        p_is_categorical=p_is_categorical,
        i_model=i_model,
        p_gotu=p_gotu,
        p_add_token=p_add_token,
        p_warmup_steps=p_warmup_steps,
        p_intermediate_size=p_intermediate_size
    )


@cli.command()
@click.option("--use-saved-params", is_flag=True, help="Use saved common parameters")
@click.option("--i-tree", required=False, type=click.Path(exists=True), help=TABLE_DESC)
@click.option(
    "--p-sequence-batch-size", default=8, show_default=True, required=False, type=int
)
@click.option(
    "--p-pairwise-batch-size", default=128, show_default=True, required=False, type=int
)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-embedding-dim", default=128, type=int)
@click.option("--p-attention-heads", default=4, type=int)
@click.option("--p-attention-layers", default=8, type=int)
@click.option("--p-intermediate-size", default=512, type=int)
@click.option(
    "--p-intermediate-activation", default="gelu", show_default=True, type=str
)
@click.option("--p-lr", default=1e-4, show_default=True, type=float)
@click.option("--p-decay-steps", default=1000, show_default=True, type=int)
@click.option("--p-max-bp", default=150, show_default=True, type=int)
@click.option("--output-dir", required=False)
@click.option("--p-weight-decay", default=0.004, show_default=True, type=float)
@click.option("--p-normalize-outputs", default=False, type=bool)
@click.option("--p-use-residual-connections", default=True, type=bool)
@click.option("--i-model", default=None, required=False, type=str)
@click.option("--p-include-bert-loss", default=True, required=False, type=bool)
@click.option("--p-use-linear-bias", default=False, type=bool)
@inject_common_params
@validate_fit_asv_encoder
@fit_asv_encoder_decorator
def fit_asv_encoder(model, **kwargs):
    print("Model has been saved to", kwargs["output_dir"])


###-------------------------------------------------------------------------------------------------------------------------------###

@cli.command()
@click.option("--use-saved-params", is_flag=True, help="Use saved common parameters")
@click.option("--i-table", required=False, type=click.Path(exists=True), help=TABLE_DESC)
@click.option("--i-tree", required=False, type=click.Path(exists=True))
@click.option(
    "--m-metadata-file",
    required=False,
    help="Metadata description",
    type=click.Path(exists=True),
)
@click.option(
    "--m-metadata-column",
    required=False,
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
@click.option("--i-unifrac-model", default=None, required=False, type=str)
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
@click.option("--output-dir", required=False)
@click.option("--p-add-token", default=False, required=False, type=bool)
@click.option("--p-gotu", default=False, required=False, type=bool)
@click.option("--p-is-categorical", default=False, required=False, type=bool)
@click.option("--p-rarefy-depth", default=5000, required=False, type=int)
@click.option("--p-weight-decay", default=0.004, show_default=True, type=float)
@click.option("--p-accumulation-steps", default=1, required=False, type=int)
@click.option("--p-unifrac-metric", default="unifrac", required=False, type=str)
@click.option("--p-loss-type", default="mse", required=False, type=str)
@click.option("--p-normalize-outputs", default=True, type=bool)
@click.option("--p-use-residual-connections", default=True, type=bool)
@click.option("--p-use-residual-pool", default=None, type=bool)
@click.option("--p-train-nuc-encoder", default=True, type=bool)
@click.option("--p-nuc-encoder", default=None)
@click.option("--p-use-linear-bias", default=False, type=bool)
@inject_common_params
@validate_fit_denoised_unifrac_regressor
@fit_denoised_unifrac_regressor_decorator
def fit_denoised_unifrac_regressor(**kwargs):
    
    print("Model has been saved to", kwargs["output_dir"])


###-------------------------------------------------------------------------------------------------------------------------------###


@cli.command()
@click.option("--use-saved-params", is_flag=True, help="Use saved common parameters")
@click.option("--i-table", required=False, type=click.Path(exists=True), help=TABLE_DESC)
@click.option("--i-taxonomy", required=False, type=click.Path(exists=True))
@click.option("--i-tax-level", default=7, type=int)
@click.option(
    "--m-metadata-file",
    required=False,
    help="Metadata description",
    type=click.Path(exists=True),
)
@click.option(
    "--m-metadata-column",
    required=False,
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
@click.option("--p-max-bp", required=False, type=int)
@click.option("--output-dir", required=False)
@click.option("--p-add-token", default=False, required=False, type=bool)
@click.option("--p-gotu", default=False, required=False, type=bool)
@click.option("--p-is-categorical", default=False, required=False, type=bool)
@click.option("--p-rarefy-depth", default=5000, required=False, type=int)
@click.option("--p-weight-decay", default=0.004, show_default=True, type=float)
@click.option("--p-accumulation-steps", default=1, required=False, type=int)
@inject_common_params
@validate_fit_taxonomy_regressor
@fit_taxonomy_regressor_decorator
def fit_taxonomy_regressor(**kwargs):
    print("Model has been saved to", kwargs["output_dir"])


###-------------------------------------------------------------------------------------------------------------------------------###


@cli.command()
@click.option("--use-saved-params", is_flag=True, help="Use saved common parameters")
@click.option(
    "--i-table",
    required=False,
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
    required=False,
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
@click.option("--output-dir", required=False, type=click.Path(exists=False))
@click.option("--p-output-dim", default=1, required=False, type=int)
@click.option("--p-add-token", default=False, required=False, type=bool)
@click.option("--p-gotu", default=False, required=False, type=bool)
@click.option("--p-is-categorical", default=False, required=False, type=bool)
@click.option("--p-rarefy-depth", default=5000, required=False, type=int)
@click.option("--p-weight-decay", default=0.004, show_default=True, type=float)
@click.option("--p-accumulation-steps", default=1, required=False, type=int)
@click.option("--p-unifrac-metric", default="unifrac", required=False, type=str)
@click.option("--p-scale-loss", default=False, type=bool)
@click.option("--p-train-nuc-encoder", default=True, type=bool)
@click.option("--p-include-count-encoder", default=True, type=bool)
@inject_common_params
@validate_fit_sample_regressor
@fit_sample_regressor_decorator
def fit_sample_regressor(**kwargs):
    print("Model has been saved to", kwargs["output_dir"])


###-------------------------------------------------------------------------------------------------------------------------------###
@cli.command()
@click.option("--use-saved-params", is_flag=True, help="Use saved common parameters")
@click.option(
    "--i-table",
    required=False,
    help=TABLE_DESC,
    type=click.Path(exists=True),
)
@click.option("--i-model-path", required=False, type=click.Path(exists=True))
@click.option(
    "--m-metadata-file",
    required=False,
    help="Metadata description",
    type=click.Path(exists=True),
)
@click.option(
    "--m-metadata-column",
    required=False,
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
@click.option("--output-dir", required=False, type=click.Path(exists=False))
@inject_common_params  
@validate_predict_sample_regressor
@predict_sample_regressor_decorator
def predict_sample_regressor(**kwargs):
    print("Predictions have been saved to", kwargs["output_dir"])


###-------------------------------------------------------------------------------------------------------------------------------###


@cli.command()
@click.option("--use-saved-params", is_flag=True, help="Use saved common parameters")
@click.option(
    "--i-asv-table",
    required=False,
    help=TABLE_DESC,
    type=click.Path(exists=True),
)
@click.option(
    "--i-gotu-table",
    required=False,
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
    required=False,
    help="Metadata description",
    type=click.Path(exists=True),
)
@click.option(
    "--m-metadata-column",
    required=False,
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
@click.option("--p-attention-layers", default=8, show_default=True, type=int)
@click.option("--p-intermediate-size", default=512, show_default=True, type=int)
@click.option(
    "--p-intermediate-activation", default="gelu", show_default=True, type=str
)
@click.option("--p-taxonomy", default=None, type=click.Path(exists=True))
@click.option("--p-taxonomy-level", default=7, show_default=True, type=int)
@click.option("--p-tree", default=None, type=click.Path(exists=True))
@click.option("--p-gen-new-table", default=True, show_default=True, type=bool)
@click.option("--p-lr", default=3e-4, show_default=True, type=float)
@click.option("--p-warmup-steps", default=0, show_default=True, type=int)
@click.option("--p-decay-steps", default=1000000, show_default=True, type=int)
@click.option("--p-max-bp", default=150, show_default=True, type=int)
@click.option("--output-dir", required=False, type=click.Path(exists=False))
@click.option("--p-output-dim", default=128, required=False, type=int)
@click.option("--p-add-token", default=False, required=False, type=bool)
@click.option("--p-is-categorical", default=False, required=False, type=bool)
@click.option("--p-gotu-rarefy-depth", default=100000, required=False, type=int)
@click.option("--p-asv-rarefy-depth", default=10000, required=False, type=int)
@click.option("--p-weight-decay", default=0.0001, show_default=True, type=float)
@click.option("--p-accumulation-steps", default=1, required=False, type=int)
@click.option("--p-unifrac-metric", default="unifrac", required=False, type=str)
@click.option("--p-scale-loss", default=False, type=bool)
@click.option("--p-normalize-outputs", default=False, type=bool)
@inject_common_params
@validate_gotu_infer
@fit_gotu_decorator
def fit_gotu(**kwargs):

    print("Model has been saved to", kwargs["output_dir"])

###-------------------------------------------------------------------------------------------------------------------------------###

def main():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    cli()


if __name__ == "__main__":
    main()
