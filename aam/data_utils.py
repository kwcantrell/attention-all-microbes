import numpy as np
import pandas as pd
import tensorflow as tf
from biom import load_table
from unifrac import unweighted


def get_table_data(table, max_bp):
    o_ids = tf.constant(table.ids(axis="observation"))

    table = table.transpose()
    data = table.matrix_data.tocoo()
    row_ind = data.row
    col_ind = data.col
    values = data.data
    indices = [[r, c] for r, c in zip(row_ind, col_ind)]
    table_data = tf.sparse.SparseTensor(
        indices=indices, values=values, dense_shape=table.shape
    )
    table_data = tf.sparse.reorder(table_data)

    table_dataset = tf.data.Dataset.from_tensor_slices(table_data)
    sequence_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=8,
        split="character",
        output_mode="int",
        output_sequence_length=max_bp,
    )
    sequence_tokenizer.adapt(o_ids[:10])

    return (o_ids, table_dataset, sequence_tokenizer)


def convert_to_normalized_dataset(values, normalize):
    if normalize == "minmax":
        shift = min(values)
        scale = max(values) - min(values)
    elif normalize == "z":
        shift = np.mean(values)
        scale = np.std(values)
    elif normalize == "none":
        shift = 0
        scale = 1
    else:
        raise Exception(f"Invalid data normalization: {normalize}")
    values_normalized = tf.expand_dims((values - shift) / scale, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(values_normalized)
    return dataset, shift, scale


def get_unifrac_dataset(table_path, tree_path):
    distance = unweighted(table_path, tree_path).data
    return tf.data.Dataset.from_tensor_slices(distance)


def batch_dataset(
    table_dataset,
    target_dataset,
    batch_size,
    shuffle=False,
    repeat=1,
    train_percent=None,
):
    dataset = tf.data.Dataset.zip(
        (tf.data.Dataset.range(table_dataset.cardinality()), table_dataset),
        target_dataset,
    )

    size = dataset.cardinality().numpy()

    if train_percent:
        size = int(size * train_percent)
        training_dataset = dataset.take(size)
    else:
        training_dataset = dataset
    training_dataset = training_dataset

    if shuffle:
        training_dataset = training_dataset.shuffle(size)
    training_dataset = training_dataset.batch(
        batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
    )

    if repeat > 1:
        training_dataset = training_dataset.repeat(repeat)

    if train_percent:
        validation_dataset = dataset.skip(size).batch(8, drop_remainder=True)
        return training_dataset, validation_dataset
    else:
        return training_dataset


def validate_metadata(table, metadata, missing_samples_flag):
    # check for mismatch samples
    ids = table.ids(axis="sample")
    shared_ids = set(ids).intersection(set(metadata.index))
    min_ids = min(len(shared_ids), len(ids), len(metadata.index))
    max_ids = max(len(shared_ids), len(ids), len(metadata.index))
    if len(shared_ids) == 0:
        raise Exception("Table and Metadata have no matching sample ids")
    if min_ids != max_ids and missing_samples_flag == "error":
        raise Exception("Table and Metadata do share all same sample ids.")
    elif min_ids != max_ids and missing_samples_flag == "ignore":
        print("Warning: Table and Metadata do share all same sample ids.")
        print("Table and metadata will be filtered")
        table = table.filter(shared_ids, inplace=False)
        metadata = metadata[metadata.index.isin(shared_ids)]
    return table, metadata


def filter_and_reorder(metadata, ids):
    metadata = metadata[metadata.index.isin(ids)]
    metadata = metadata.reindex(ids)
    return metadata


def shuffle_table(table):
    ids = table.ids(axis="sample")
    np.random.shuffle(ids)
    return table.sort_order(ids)


def extract_col(metadata, col, output_dtype=None):
    metadata_col = metadata[col]
    if output_dtype is not None:
        metadata_col = metadata_col.astype(output_dtype)
    return metadata_col


def load_data(
    table_path,
    max_bp,
    batch_size,
    repeat=1,
    shuffle_samples=False,
    train_percent=0.9,
    tree_path=None,
    metadata_path=None,
    metadata_col=None,
    missing_samples_flag=None,
):
    table = load_table(table_path)
    if shuffle_samples:
        table = shuffle_table(table)

    if tree_path:
        sample_ids = table.ids(axis="sample")
        o_ids, table_dataset, sequence_tokenizer = get_table_data(table, max_bp)
        unifrac_dataset = get_unifrac_dataset(table_path, tree_path)
        training_dataset, validation_dataset = batch_dataset(
            table_dataset,
            unifrac_dataset,
            batch_size,
            shuffle=shuffle_samples,
            repeat=repeat,
            train_percent=train_percent,
        )
        return {
            "table": table,
            "sample_ids": sample_ids,
            "o_ids": o_ids,
            "sequence_tokenizer": sequence_tokenizer,
            "unifrac_dataset": unifrac_dataset,
            "training_dataset": training_dataset,
            "validation_dataset": validation_dataset,
            "mean": 0,
            "std": 1,
        }

    if metadata_path:
        metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)
        table, metadata = validate_metadata(table, metadata, missing_samples_flag)
        sample_ids = table.ids(axis="sample")
        metadata = filter_and_reorder(metadata, sample_ids)
        o_ids, table_dataset, sequence_tokenizer = get_table_data(table.copy(), max_bp)

        regression_data = extract_col(metadata, metadata_col, output_dtype=np.float32)
        regression_dataset, mean, std = convert_to_normalized_dataset(
            regression_data, "z"
        )

        training_dataset, validation_dataset = batch_dataset(
            table_dataset,
            regression_dataset,
            batch_size,
            shuffle=shuffle_samples,
            repeat=repeat,
            train_percent=train_percent,
        )

        return {
            "table": table,
            "sample_ids": sample_ids,
            "o_ids": o_ids,
            "sequence_tokenizer": sequence_tokenizer,
            "metadata": metadata,
            "training_dataset": training_dataset,
            "validation_dataset": validation_dataset,
            "mean": mean,
            "std": std,
        }
