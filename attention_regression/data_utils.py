import numpy as np
import tensorflow as tf
from biom import load_table


def load_biom_table(fp):
    table = load_table(fp)
    return table


def shuffle_table(table):
    ids = table.ids(axis='sample')
    np.random.shuffle(ids)
    return table.sort_order(ids)


def convert_table_to_dataset(table):
    o_ids = tf.constant(table.ids(axis='observation'))
    table = table.transpose()
    table_coo = table.matrix_data.tocoo()
    row_ind = table_coo.row
    col_ind = table_coo.col
    values = table_coo.data
    indices = [[r, c] for r, c in zip(row_ind, col_ind)]
    sparse_tensor = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=table.shape
    )
    sparse_tensor = tf.sparse.reorder(sparse_tensor)

    def get_inputs(x):
        return (
            tf.squeeze(tf.gather(o_ids, x.indices)),
            tf.cast(x.values, dtype=tf.float32)
        )

    return (
        tf.data.Dataset
        .from_tensor_slices(sparse_tensor)
        .map(get_inputs, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


def convert_to_normalized_dataset(values, normalize):
    if normalize == 'minmax':
        shift = min(values)
        scale = max(values) - min(values)
    elif normalize == 'z':
        shift = np.mean(values)
        scale = np.std(values)
    elif normalize == 'none':
        shift = 1
        scale = 1
    else:
        raise Exception(f"Invalid data normalization: {normalize}")
    values_normalized = (values - shift) / scale
    dataset = tf.data.Dataset.from_tensor_slices(values_normalized)
    return dataset, shift, scale


def convert_to_categorical_dataset(values):
    dataset = tf.data.Dataset.from_tensor_slices(values)
    return dataset


def filter_and_reorder(metadata, ids):
    metadata = metadata[metadata.index.isin(ids)]
    metadata = metadata.reindex(ids)
    return metadata


def extract_col(metadata, col, output_dtype=None):
    metadata_col = metadata[col]
    if output_dtype is not None:
        metadata_col = metadata_col.astype(output_dtype)
    return metadata_col


def tokenize_dataset(dataset, vocab):
    sequence_tokenizer = tf.keras.layers.StringLookup(
        vocabulary=vocab,
        output_mode='int'
    )
    tokenized_ids = sequence_tokenizer(dataset.map(lambda ids, _: ids))
    rclr_values = dataset.map(lambda _, rclr_values: rclr_values)
    return tf.data.Dataset.zip((tokenized_ids, rclr_values))


def batch_dataset(dataset, batch_size, repeat=None, shuffle=False):
    def extract_zip(feature_rclr, target):
        gx = tf.exp(tf.reduce_mean(tf.math.log(feature_rclr[1])))
        inputs = {
            "feature": feature_rclr[0],
            "rclr": tf.math.log(
                feature_rclr[1] / gx,
            )
        }
        outputs = {"reg_out": target}
        return (inputs, outputs)

    if shuffle:
        size = dataset.cardinality()
        dataset = dataset.shuffle(size, reshuffle_each_iteration=True)

    if repeat:
        dataset = dataset.repeat(repeat)

    input_pad = {
        "feature": [None],
        "rclr": [None]
    }
    input_pad_val = {
        "feature": "<MASK>",
        "rclr": tf.cast(0.0, dtype=tf.float32)
    }

    output_pad = {"reg_out": []}
    output_pad_val = {
        "reg_out": tf.cast(0.0, dtype=tf.float32)
    }

    dataset = (
        dataset
        .map(extract_zip)
        .padded_batch(
            batch_size,
            padded_shapes=(input_pad, output_pad),
            padding_values=(input_pad_val, output_pad_val),
            drop_remainder=False
        )
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def train_val_split(
    dataset: tf.data.Dataset,
    train_percent: float
):
    size = dataset.cardinality().numpy()
    train_size = int(size*train_percent)
    training_dataset = (
        dataset
        .take(train_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_dataset = (
        dataset
        .skip(train_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    return training_dataset, validation_dataset
