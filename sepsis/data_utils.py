import numpy as np
import tensorflow as tf
from biom import load_table, Table
from biom.util import biom_open
from gemelli.preprocessing import rclr_transformation


def load_biom_table(fp):
    table = load_table(fp)
    return table


def shuffle_table(table):
    ids = table.ids(axis='sample')
    np.random.shuffle(ids)
    return table.sort_order(ids)


def create_rclr_table(table_fp, rclr_table_fp):
    table = load_biom_table(table_fp)
    rclr_table = rclr_transformation(table)
    rclr_matrix = rclr_table.matrix_data.toarray()
    rclr_matrix[~np.isfinite(rclr_matrix)] = 0.0
    rclr_table = Table(rclr_matrix,
                       rclr_table.ids(axis='observation'),
                       rclr_table.ids(axis='sample'))
    with biom_open(rclr_table_fp, 'w') as f:
        rclr_table.to_hdf5(f, generated_by="gemelli.preprocessing.rclr_transformation")


def convert_table_to_dataset(table):
    o_ids = tf.constant(table.ids(axis='observation'))
    table = table.transpose()
    table_coo = table.matrix_data.tocoo()
    row_ind = table_coo.row
    col_ind = table_coo.col
    values = table_coo.data
    indices = [[r, c] for r, c in zip(row_ind, col_ind)]
    sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=table.shape)
    sparse_tensor = tf.sparse.reorder(sparse_tensor)

    def get_inputs(x):
        return (tf.squeeze(tf.gather(o_ids, x.indices)),
                tf.cast(x.values, dtype=tf.float32))

    return (tf.data.Dataset.from_tensor_slices(sparse_tensor)
            .map(get_inputs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
            )


def convert_to_normalized_dataset(values):
    mean = values.mean()
    std = values.std()
    values_normalized = (values - mean) / std
    dataset = tf.data.Dataset.from_tensor_slices(values_normalized)
    return dataset, mean, std


def filter_and_reorder(metadata, ids):
    metadata = metadata[metadata.index.isin(ids)]
    metadata = metadata.reindex(ids)
    return metadata


def extract_col(metadata, col, output_dtype=None):
    metadata_col = metadata['host_age']
    if output_dtype is not None:
        metadata_col = metadata_col.astype(output_dtype)
    return metadata_col


def tokenize_dataset(dataset, vocab):
    sequence_tokenizer = tf.keras.layers.StringLookup(
        vocabulary=vocab,
        output_mode='int')
    tokenized_ids = sequence_tokenizer(dataset.map(lambda ids, _: ids))
    rclr_values = dataset.map(lambda _, rclr_values: rclr_values)
    return tf.data.Dataset.zip((tokenized_ids, rclr_values))


def batch_dataset(dataset, batch_size, shuffle=False):
    def extract_zip(feature_rclr, target):
        return ({
            "feature": feature_rclr[0],
            "rclr": feature_rclr[1]},
             target)

    if shuffle:
        size = dataset.cardinality()
        dataset = dataset.shuffle(size, reshuffle_each_iteration=True)
    dataset = (dataset
               .map(extract_zip)
               .padded_batch(batch_size,
                             padded_shapes=({"feature": [None],
                                             "rclr": [None]}, []),
                             padding_values=({"feature": "<MASK>",
                                             "rclr": tf.cast(0.0, dtype=tf.float32)},
                                             tf.cast(0.0, dtype=tf.float32)),
                             drop_remainder=False)
               .prefetch(tf.data.AUTOTUNE))
    return dataset.prefetch(tf.data.AUTOTUNE)


def train_val_split(dataset: tf.data.Dataset,
                    train_percent: float):
    size = dataset.cardinality().numpy()
    train_size = int(size*train_percent)
    training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
    return training_dataset, validation_dataset