import os
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn

# from tensorboard.plugins import projector
from aam.losses import _pairwise_distances
from skbio.stats.distance import DistanceMatrix
import skbio.stats.ordination
from unifrac import unweighted
from biom import load_table
from aam.data_utils import (
    get_sequencing_dataset,
    get_unifrac_dataset,
    combine_datasets,
    batch_dataset,
)

pred_pcoa = skbio.stats.ordination.pcoa(
    pred_unifrac_distances, method="fsvd", number_of_dimensions=3, inplace=True
)
pred_pcoa.write(os.path.join(output_dir, "pred_pcoa.pcoa"))
true_unifrac_distances = unweighted(i_table, i_tree).filter(
    table.ids(axis="sample")[sample_indices]
)
true_pcoa = skbio.stats.ordination.pcoa(
    true_unifrac_distances, method="fsvd", number_of_dimensions=3, inplace=True
)
true_pcoa.write(os.path.join(output_dir, "true_pcoa.pcoa"))
