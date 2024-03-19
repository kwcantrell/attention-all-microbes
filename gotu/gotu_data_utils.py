import numpy as np
import tensorflow as tf
from biom import load_table
NUM_GOTUS = 6838

def load_biom_table(fp):
    table = load_table(fp)
    return table

def convert_to_tf_dataset(data):
    o_ids = tf.constant(data.ids(axis='observation'))
    data = data.transpose()
    table = data.matrix_data.tocoo()
    row_ind = table.row
    col_ind = table.col
    values = table.data
    indices = [[r, c] for r, c in zip(row_ind, col_ind)]
    sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=data.shape)
    sparse_tensor = tf.sparse.reorder(sparse_tensor)
    get_id = lambda x: tf.gather(o_ids, x.indices)
    
    return (tf.data.Dataset.from_tensor_slices(sparse_tensor)
            .map(get_id, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
            )

def asv_encode_and_convert(data):
    asv_dataset = convert_to_tf_dataset(data)
    sequence_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=5,
        split='character',
        output_mode='int',
        output_sequence_length=150)
    sequence_tokenizer.adapt(asv_dataset.take(1))
    asv_dataset = asv_dataset.map(lambda x: sequence_tokenizer(x))

    return asv_dataset
    
def gotu_encode_and_convert(data):
    gotu_dataset = convert_to_tf_dataset(data)
    o_ids = tf.data.Dataset.from_tensor_slices(data.ids(axis='observation'))
    sequence_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=o_ids.cardinality().numpy() + 1, # len of gotus + 1 for padding
        output_mode='int',
        output_sequence_length=1)

    sequence_tokenizer.adapt(o_ids)
    gotu_list = data.ids(axis='observation')
    gotu_tokens = sequence_tokenizer(gotu_list) + 2
    print("!!!!!!!!!!!!!")
    print(gotu_list.shape)
    print(gotu_tokens.numpy().shape)
    gotu_dict = {
        t: gotu for t, gotu in zip(list(tf.squeeze(gotu_tokens, axis=-1).numpy()), gotu_list)
    }
    
    gotu_dataset = gotu_dataset.map(lambda x:(
        tf.concat(
            [tf.constant([[1]], dtype=tf.int64), sequence_tokenizer(x) + 2, tf.constant([[2]], dtype=tf.int64)],
            axis=0)
        ))
    return gotu_dataset, gotu_dict

def combine_all_datasets(asv_dataset, gotu_dataset):
    dataset_size = asv_dataset.cardinality()
    return ((tf.data.Dataset
            .zip(asv_dataset, gotu_dataset)
            .shuffle(dataset_size, reshuffle_each_iteration=False)    
            .prefetch(tf.data.AUTOTUNE)
            ))

def batch_dataset(dataset, batch_size):
    def extract_zip(asv_data, gotu_data):
        return ({
            "encoder_inputs": asv_data,
            "decoder_inputs": gotu_data[:-1, :],
        },
        tf.squeeze(gotu_data[1:, :], axis=-1)
                )


    dataset = dataset.cache()
    size = dataset.cardinality()
    dataset = dataset.shuffle(size, reshuffle_each_iteration=True)
    dataset = (dataset
                .map(extract_zip)
                .padded_batch(batch_size,
                             padded_shapes=({"encoder_inputs": [None, 150],
                                             "decoder_inputs": [None, 1]},
                                           [None]),
                             drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    # dataset = (dataset
    #     .padded_batch(batch_size, padded_shapes=([None,150], [NUM_GOTUS + 1, 1]), drop_remainder=True)
    #     .prefetch(tf.data.AUTOTUNE)
    # )

    dataset = dataset.cache()
    return dataset.prefetch(tf.data.AUTOTUNE)

def generate_gotu_dataset(gotu_fp, asv_fp):
    gotu_encoded, gotu_dict = gotu_encode_and_convert(load_biom_table(gotu_fp))
    asv_encoded = asv_encode_and_convert(load_biom_table(asv_fp))
    
    return combine_all_datasets(asv_encoded, gotu_encoded), gotu_dict

