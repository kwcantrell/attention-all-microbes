import numpy as np
import pandas as pd
import tensorflow as tf
from biom import load_table
from unifrac import unweighted


def align_table_and_metadata(table_path,
                             metadata_path,
                             metadata_col=None,
                             is_regressor=True):
    metadata = pd.read_csv(metadata_path, sep='\t', index_col=0)
    if is_regressor:
        metadata = metadata[pd.to_numeric(metadata[metadata_col],
                                          errors='coerce').notnull()]
        metadata[metadata_col] = metadata[metadata_col].astype(np.float32)
    else:
        metadata[metadata_col] = metadata[metadata_col].astype('category')
        metadata[metadata_col] = metadata[metadata_col].cat.codes.astype('int')
    table = load_table(table_path)
    return table.align_to_dataframe(metadata, axis='sample')


def get_sequencing_dataset(table_path):
    if isinstance(table_path, str):
        table = load_table(table_path)
    else:
        table = table_path
    o_ids = tf.constant(table.ids(axis='observation'))
    table = table.transpose()
    data = table.matrix_data.tocoo()
    row_ind = data.row
    col_ind = data.col
    values = data.data
    indices = [[r, c] for r, c in zip(row_ind, col_ind)]
    table_data = tf.sparse.SparseTensor(indices=indices, values=values,
                                        dense_shape=table.shape)
    table_data = tf.sparse.reorder(table_data)

    def get_asv_id(x):
        return tf.gather(o_ids, x.indices)

    return (tf.data.Dataset.from_tensor_slices(table_data)
            .map(get_asv_id,
                 num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


def get_unifrac_dataset(table_path, tree_path):
    distance = unweighted(table_path, tree_path).data
    return (tf.data.Dataset.from_tensor_slices(distance)
            .prefetch(tf.data.AUTOTUNE))


def combine_datasets(seq_dataset, dist_dataset, max_bp, add_index=False):
    sequence_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=7,
        split='character',
        output_mode='int',
        output_sequence_length=max_bp)
    sequence_tokenizer.adapt(seq_dataset.take(1))

    seq_dataset = seq_dataset.map(lambda x: sequence_tokenizer(x))
    dataset_size = seq_dataset.cardinality()

    if add_index:
        zip = (tf.data.Dataset.range(dataset_size),
               seq_dataset,
               dist_dataset)
    else:
        zip = (seq_dataset,
               dist_dataset)
    return (tf.data.Dataset
            .zip(*zip)
            .prefetch(tf.data.AUTOTUNE))


def batch_dataset(dataset,
                  batch_size,
                  shuffle=False,
                  repeat=1,
                  is_pairwise=False):
    dataset = dataset.cache()
    size = dataset.cardinality()

    if shuffle:
        dataset = dataset.shuffle(size, reshuffle_each_iteration=True)

    if is_pairwise:
        def extract_zip(ind, seq, dist):
            return (seq, tf.gather(dist, ind, axis=1, batch_dims=0))

        def step_pad(ind, seq, dist):
            ASV_DIM = 0
            FACTOR = 32
            shape = tf.shape(seq)[ASV_DIM]
            pad = shape // FACTOR * FACTOR + FACTOR - shape
            return (ind, tf.pad(seq, [[0, pad], [0, 0]]), dist)

        padded_shape = ([], [None, 100], [None])
    else:
        def extract_zip(seq, y):
            return (seq, y)

        def step_pad(seq, y):
            ASV_DIM = 0
            shape = tf.shape(seq)[ASV_DIM]
            pad = shape // 8 * 8 + 8 - shape
            return (tf.pad(seq, [[0, pad], [0, 0]]), y)

        padded_shape = ([None, 100], [])

    dataset = (dataset
               .map(step_pad, num_parallel_calls=tf.data.AUTOTUNE)
               .padded_batch(batch_size,
                             padded_shapes=padded_shape,
                             drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE)
               .map(extract_zip)
               .prefetch(tf.data.AUTOTUNE))

    dataset = dataset.repeat(repeat)
    if not shuffle:
        dataset = dataset.cache()

    return dataset.prefetch(tf.data.AUTOTUNE)
