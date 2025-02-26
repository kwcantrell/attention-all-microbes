import sys

import numpy as np
import pandas as pd
import skbio
import tensorflow as tf
from skbio.stats.distance import DistanceMatrix
from sklearn.model_selection import StratifiedKFold

from aam.data_handlers.generator_dataset_v2 import (
    GeneratorDatasetV2,
    get_dataset,
)
from aam.models.triplet_encoder import TripletEncoder

tf.keras.mixed_precision.set_global_policy("mixed_float16")
gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

metadata_file = sys.argv[1]
metadata_col = sys.argv[2]
model_path = sys.argv[3]
output = sys.argv[4]
taxonomy = pd.read_csv("agp-unique-samples-taxonomy.tsv", sep="\t", index_col=0)

metadata = pd.read_csv(metadata_file, sep="\t", index_col=0, dtype={"#SampleID": str})

groups = []
for group, df in metadata.groupby(metadata_col):
    if df.shape[0] >= 10:
        groups.append(df)
metadata = pd.concat(groups)
metadata = metadata.groupby(metadata_col).sample(10, random_state=42, replace=False)
print(metadata.shape, metadata[metadata_col].value_counts())
model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded!!")
ug = GeneratorDatasetV2(
    table="agp-unique-samples.biom",
    metadata=metadata,
    metadata_column=metadata_col,
    taxonomy=taxonomy,
    gen_new_tables=False,
    shuffle=True,
    epochs=1,
    return_sample_ids=True,
    is_categorical=True,
    batch_size=8,
    rarefy_depth=5000,
)
dataset = get_dataset(ug)


def compute_sample_distances(x, y=None):
    if y is None:
        y = x
    sample_distances = np.square(x[np.newaxis, :, :] - y[:, np.newaxis, :])
    sample_distances = np.sum(sample_distances, axis=-1)
    sample_distances = np.sqrt(sample_distances)
    return sample_distances


sample_encodings, ys = model.predict(dataset.take(ug.steps_per_epoch))
sample_encodings = sample_encodings.astype(np.float32)
sample_ids = [s.decode("utf-8") for s in ys]
ug.stop()

sample_distances = compute_sample_distances(sample_encodings)
num_samples = sample_distances.shape[0]
distances = DistanceMatrix(sample_distances, sample_ids, validate=False)

distances.write(f"{output}.dist")
pcoa = skbio.stats.ordination.pcoa(
    distances, method="fsvd", number_of_dimensions=5, inplace=True
)
pcoa.write(f"{output}.pcoa")


group_centers = {}
metadata = metadata.loc[metadata.index.isin(sample_ids)]
for group, group_df in metadata.groupby(metadata_col):
    group_samples = group_df.index.to_list()
    group_indices = [i for i, s_id in enumerate(sample_ids) if s_id in group_samples]

    group_center = np.mean(sample_encodings[group_indices], axis=0)
    group_centers[group] = {"indices": group_indices, "center": group_center}


def _dot_prod(x, y=None):
    if y is None:
        y = x
    return np.sum(x * y, axis=-1)


def _proj(u, v):
    u_norms = np.sqrt(np.sum(u * u, axis=-1))
    u /= u_norms
    return v - _dot_prod(v, u) * u


group_centers
groups = list(group_centers.keys())
anchor_group = groups[0]
anchor_group_center = group_centers[anchor_group]["center"]
anchor_group_unit_center = anchor_group_center
anchor_group_indices = group_centers[anchor_group]["indices"]
anchor_group_unit_center = anchor_group_unit_center / np.sqrt(
    np.sum(_dot_prod(anchor_group_unit_center))
)
for group in groups[1:]:
    anchor_points = sample_encodings[group_centers[anchor_group]["indices"]]
    group_points = sample_encodings[group_centers[group]["indices"]]
    group_distances = compute_sample_distances(group_points, anchor_points)
    closest_anchors = np.argmin(group_distances, axis=-1)
    group_projections = _proj(anchor_points[closest_anchors], group_points)
    sample_encodings[group_centers[group]["indices"]] = group_projections
    group_centers[anchor_group]["indices"] += group_centers[group]["indices"]

    # group_center = group_centers[group]["center"]
    # group_indices = group_centers[group]["indices"]
    # orthogal_direction = (
    #     _dot_prod(group_center, anchor_group_unit_center) * anchor_group_unit_center
    # )
    # group_shift = group_center - orthogal_direction
    # sample_encodings[group_indices] -= group_shift
# sample_encodings -= anchor_group_center

sample_distances = compute_sample_distances(sample_encodings)
num_samples = sample_distances.shape[0]
distances = DistanceMatrix(sample_distances, sample_ids, validate=False)

distances.write(f"{output}-normalized.dist")
pcoa = skbio.stats.ordination.pcoa(
    distances, method="fsvd", number_of_dimensions=5, inplace=False
)
print(pcoa.proportion_explained)
pcoa.write(f"{output}-normalized.pcoa")
