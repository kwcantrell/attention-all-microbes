"""gotu.py"""

import os
import time
import click


from gotu.gotu_data_utils import (
    create_training_dataset,
    create_prediction_dataset,
    save_dataset,
    save_gotu_dict,
    load_dataset,
    load_model_dict,
)
from gotu.gotu_model import (
    gotu_classification,
    gotu_predict,
    run_gotu_predictions,
    run_gotu_training
    )

# TODO: Add catch statements to cli to validate file/fp's


@click.command(help="train ASV to GOTU model")
@click.argument("dataset-path", type=click.Path(exists=True))
@click.option("--model_fp", type=click.Path(), help="File path to save the model.")
def run_gotu_training_cli(dataset_path, model_fp):
    training_fp = f"{dataset_path}/training_set"
    val_fp = f"{dataset_path}/validation_set"
    training_dataset = load_dataset(training_fp)
    validation_dataset = load_dataset(val_fp)
    run_gotu_training(
        training_dataset, validation_dataset, model_fp=model_fp
    )


@click.command(help="Run predictions with previously trained models")
@click.argument("asv_fp", type=click.Path(exists=True))
@click.argument("gotu_fp", type=click.Path(exists=True))
@click.argument("model_fp", type=click.Path(exists=True))
@click.argument("pred_out_path", type=click.Path())
def run_gotu_predictions_cli(asv_fp, gotu_fp, model_fp, pred_out_path):
    run_gotu_predictions(asv_fp, gotu_fp, model_fp, pred_out_path)


@click.command(help="Creates tokenized training/validation datasets")
@click.argument("asv-fp", type=click.Path(exists=True))
@click.argument("gotu-fp", type=click.Path(exists=True))
@click.option("--name", default="a2g")
@click.option("--train-split", default=0.8, show_default=True)
@click.option("--outpath", type=click.Path(exists=False))
def create_training_set_cli(asv_fp, gotu_fp, name, train_split, outpath):
    timestamp = time.strftime("%Y%m%d-%H%M")
    full_dir = f"{outpath}/{timestamp}-{name}"

    try:
        os.mkdir(full_dir)
    except NotADirectoryError as e:
        print(f"Failed to create out folder directory: {e}")

    training_fp = f"training_set"
    val_fp = f"validation_set"
    dict_fp = f"gotu_dict"
    training_batched, val_batched, gotu_dict = create_training_dataset(
        asv_fp, gotu_fp, train_split
    )
    save_dataset(training_batched, full_dir, training_fp)
    save_dataset(val_batched, full_dir, val_fp)
    save_gotu_dict(gotu_dict, full_dir, dict_fp)

@click.command(help="Creates tokenized dataset for running predictions")
@click.argument("asv-fp", type=click.Path(exists=True))
@click.argument("gotu-fp", type=click.Path(exists=True))
@click.option("--name", default="a2g")
@click.option("--outpath", type=click.Path(exists=False))
def create_training_set_cli(asv_fp, gotu_fp, name, train_split, outpath):
    timestamp = time.strftime("%Y%m%d-%H%M")
    full_dir = f"{outpath}/{timestamp}-{name}"

    try:
        os.mkdir(full_dir)
    except NotADirectoryError as e:
        print(f"Failed to create out folder directory: {e}")
    
    dataset_fp = f"pred-dataset"
    dict_fp = f"gotu_dict"
    training_batched, val_batched, gotu_dict = create_training_dataset(
        asv_fp, gotu_fp, train_split
    )
    save_dataset(training_batched, full_dir, dataset_fp)
    save_gotu_dict(gotu_dict, full_dir, dict_fp)
    
    
@click.group()
def cli():
    pass


cli.add_command(run_gotu_training_cli, "train")
cli.add_command(run_gotu_predictions_cli, "predict")
cli.add_command(create_training_set_cli, "format-training-data")

if __name__ == "__main__":
    cli()