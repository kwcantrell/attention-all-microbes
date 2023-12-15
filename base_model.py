import click
import os
import json
import numpy as np
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from amplicon_gpt.model_utils import load_full_base_model
from amplicon_gpt.data_utils import create_base_sequencing_data
from amplicon_gpt.losses import _pairwise_distances
from unifrac import unweighted
from biom import load_table
from skbio.stats.distance import DistanceMatrix
import skbio.stats.ordination

@click.group('base_model')
@click.pass_context
def base_model(ctx):
    pass

@base_model.command('unifrac_distance')
@click.pass_context
@click.option('--config-json', type=click.Path(exists=True))
def unifrac_distance(ctx, config_json):
    with open(config_json) as f:
        config = json.load(f)
    table = load_table(config['table_path'])
    dataset = create_base_sequencing_data(**config)
    base_model = load_full_base_model(**config)
    total_samples = int(table.shape[1] / config['batch_size']) * config['batch_size']
    
    sample_indices = np.arange(total_samples)
    np.random.shuffle(sample_indices)
    sample_indices = sample_indices[:config['num_samples']]
    pred = base_model.predict(dataset)

    pred = tf.constant(pred[sample_indices])
    distances = _pairwise_distances(pred, squared=False)
    pred_unifrac_distances = DistanceMatrix(distances.numpy(), table.ids(axis='sample')[sample_indices], validate=False)
    pred_pcoa = skbio.stats.ordination.pcoa(pred_unifrac_distances, method='eigh', inplace=False)
    pred_pcoa.write(config['pred_pcoa_path'])

    true_unifrac_distances = unweighted(config['table_path'], config['tree_path']).filter(table.ids(axis='sample')[sample_indices])
    true_pcoa = skbio.stats.ordination.pcoa(true_unifrac_distances, method='eigh', inplace=False)
    true_pcoa.write(config['true_pcoa_path'])

def main():
    base_model(prog_name='base_model')

if __name__ == '__main__':
    main()