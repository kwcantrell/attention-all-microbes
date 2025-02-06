from __future__ import annotations

import datetime
import os
from typing import Union

import click
import numpy as np
import pandas as pd
import tensorflow as tf
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

from attention_wrappers import fit_asv_encoder_decorator, fit_denoised_unifrac_regressor_decorator, fit_taxonomy_regressor_decorator
from attention_wrappers import fit_sample_regressor_decorator, predict_sample_regressor_decorator, fit_gotu_decorator
import pytest
from click.testing import CliRunner

@click.group()
class cli:
    pass


TABLE_DESC = "Feature table containing all features that should be used for target prediction."
TEST_SIZE_DESC = "Fraction of input samples to exclude from training set and use for classifier testing."
CV_DESC = "Number of k-fold cross-validations to perform."
STRAT_DESC = "Evenly stratify training and test data among metadata categories. If True, all values in column must match at least two samples."
MISSING_SAMP_DESC = 'How to handle missing samples in metadata. "error" will fail if missing samples are detected. "ignore" will cause the feature table and metadata to be filtered, so that only samples found in both files are retained.'


GLOBAL_CONFIGURATIONS = {}


@cli.command()
@click.option("--i-tree", required=True, type=click.Path(exists=True), help=TABLE_DESC)
@click.option("--p-sequence-batch-size", default=8, show_default=True, required=False, type=int)
@click.option("--p-pairwise-batch-size", default=128, show_default=True, required=False, type=int)
@click.option("--p-epochs", default=1000, show_default=True, type=int)
@click.option("--p-dropout", default=0.0, show_default=True, type=float)
@click.option("--p-embedding-dim", default=128, type=int)
@click.option("--p-attention-heads", default=4, type=int)
@click.option("--p-attention-layers", default=8, type=int)
@click.option("--p-intermediate-size", default=512, type=int)
@click.option("--p-intermediate-activation", default="gelu", show_default=True, type=str)
@click.option("--p-lr", default=1e-4, show_default=True, type=float)
@click.option("--p-decay-steps", default=1000, show_default=True, type=int)
@click.option("--p-max-bp", default=150, show_default=True, type=int)
@click.option("--output-dir", required=True)
@click.option("--p-weight-decay", default=0.004, show_default=True, type=float)
@click.option("--p-normalize-outputs", default=False, type=bool)
@click.option("--p-use-residual-connections", default=True, type=bool)
@click.option("--i-model", default=None, required=False, type=str)
@click.option("--p-include-bert-loss", default=True, required=False, type=bool)
@click.option("--p-use-linear-bias", default=False, type=bool)
@fit_asv_encoder_decorator
def fit_asv_encoder(model, **kwargs):
    print("Model has been saved to", kwargs["output_dir"])


###-------------------------------------------------------------------------------------------------------------------------------###
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
@click.option("--i-unifrac-model", default=None, required=False, type=str)
@click.option("--p-embedding-dim", default=128, type=int)
@click.option("--p-attention-heads", default=4, type=int)
@click.option("--p-attention-layers", default=4, type=int)
@click.option("--p-intermediate-size", default=1024, type=int)
@click.option("--p-intermediate-activation", default="relu", show_default=True, type=str)
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
@click.option("--p-loss-type", default="mse", required=False, type=str)
@click.option("--p-normalize-outputs", default=True, type=bool)
@click.option("--p-use-residual-connections", default=True, type=bool)
@click.option("--p-use-residual-pool", default=None, type=bool)
@click.option("--p-train-nuc-encoder", default=True, type=bool)
@click.option("--p-nuc-encoder", default=None)
@click.option("--p-use-linear-bias", default=False, type=bool)
@fit_denoised_unifrac_regressor_decorator
def fit_denoised_unifrac_regressor(**kwargs):
    
    print("Model has been saved to", kwargs["output_dir"])

###-------------------------------------------------------------------------------------------------------------------------------###


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
@click.option("--p-intermediate-activation", default="relu", show_default=True, type=str)
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
@fit_taxonomy_regressor_decorator
def fit_taxonomy_regressor(**kwargs):
    print("Model has been saved to", kwargs["output_dir"])

###-------------------------------------------------------------------------------------------------------------------------------###

@cli.command()
@click.option(
    "--i-table",
    required=True,
    help=TABLE_DESC,
    type=click.Path(exists=True),
)
@click.option("--i-base-model-path", default=None, required=False, type=click.Path(exists=True))
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
@click.option("--p-intermediate-activation", default="relu", show_default=True, type=str)
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
@click.option("--p-train-nuc-encoder", default=True, type=bool)
@click.option("--p-include-count-encoder", default=True, type=bool)
@fit_sample_regressor_decorator
def fit_sample_regressor(**kwargs):
    print("Model has been saved to", kwargs["output_dir"])

###-------------------------------------------------------------------------------------------------------------------------------###
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
@click.option("--p-mixed-precision / --p-no-mixed-precision", default=True, required=False)
@click.option("--output-dir", required=True, type=click.Path(exists=False))
@predict_sample_regressor_decorator
def predict_sample_regressor(**kwargs):
    print("Predictions have been saved to", kwargs["output_dir"])

###-------------------------------------------------------------------------------------------------------------------------------###

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
@click.option("--i-base-model-path", default=None, required=False, type=click.Path(exists=True))
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
@click.option("--p-attention-layers", default=8, show_default=True, type=int)
@click.option("--p-intermediate-size", default=512, show_default=True, type=int)
@click.option("--p-intermediate-activation", default="gelu", show_default=True, type=str)
@click.option("--p-taxonomy", default=None, type=click.Path(exists=True))
@click.option("--p-taxonomy-level", default=7, show_default=True, type=int)
@click.option("--p-tree", default=None, type=click.Path(exists=True))
@click.option("--p-gen-new-table", default=True, show_default=True, type=bool)
@click.option("--p-lr", default=3e-4, show_default=True, type=float)
@click.option("--p-warmup-steps", default=0, show_default=True, type=int)
@click.option("--p-decay-steps", default=1000000, show_default=True, type=int)
@click.option("--p-max-bp", default=150, show_default=True, type=int)
@click.option("--output-dir", required=True, type=click.Path(exists=False))
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
