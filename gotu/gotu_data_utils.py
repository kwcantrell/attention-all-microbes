"""gotu_data_utils.py"""

import json

import numpy as np
import tensorflow as tf
from biom import load_table

NUM_GOTUS = 6838

def save_gotu_dict(gotu_dict: dict, outpath: str, name: str):
    """
    Converts integer keys of a dictionary to strings and saves the dictionary as a JSON file.

    Args:
    - gotu_dict (dict): The dictionary with integer keys and corresponding values.
    - outpath (str): The directory path where the JSON file will be saved.
    - name (str): The name of the JSON file to be saved (without the extension).
    """
    # Convert integer keys to string keys
    str_keys_gotu_dict = {str(k): v for k, v in gotu_dict.items()}

    # Construct full path for the JSON file
    full_path = f"{outpath}/{name}.json"

    # Write the dictionary with string keys to a JSON file
    with open(full_path, 'w') as file:
        json.dump(str_keys_gotu_dict, file, indent=4)

    print(f"Dictionary saved as {full_path}")
    

def save_dataset(dataset: tf.data.Dataset, out_path: str, name: str):
    """
    Saves a TensorFlow dataset to a specified path with an optional timestamped name.

    Parameters:
    - dataset (tf.data.Dataset): The TensorFlow dataset to be saved.
    - out_path (str): The output directory where the dataset will be saved.
    - name (str): The base name for the saved file.
    """
    full_path = f"{out_path}/{name}"
    try:
        dataset.save(full_path)
        print(f"Saved {full_path}")
    except Exception as e:
        print(f"Failed to save dataset: {e}")


def load_dataset(dataset_path: str, compression=None) -> tf.data.Dataset:
    
    tf_dataset = tf.data.Dataset.load(dataset_path, compression=compression)
    """
    Loads a previously saved dataset for training

    Parameters:
    - dataset_path (str): Path to tensorflow dataset.
    - compression (str) : Either 'gzip' or None. Defaults to None
    
    Returns:
        tf_dataset (tf.data.Dataset): Loaded tf.dataset
    """
    return tf_dataset


def load_model_dict(file_path):
    """
    Load a JSON file and convert all dictionary keys to integers.

    Args:
        file_path (str): The path to the JSON file to be read.

    Returns:
        dict: A dictionary with all keys converted to integers where possible. Keys that cannot be converted to integers are omitted.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON file.
        ValueError: If a key is not a valid string representation of an integer and is thus skipped.

    """
    with open(file_path, 'r') as json_file:
        data_dict = json.load(json_file)
        data_dict = {int(k): v for k, v in data_dict.items()}

    return data_dict


def load_biom_table(fp):
    table = load_table(fp)
    return table


def convert_to_tf_dataset(data):
    o_ids = tf.constant(data.ids(axis="observation"))
    data = data.transpose()
    table = data.matrix_data.tocoo()
    row_ind = table.row
    col_ind = table.col
    values = table.data
    indices = [[r, c] for r, c in zip(row_ind, col_ind)]
    sparse_tensor = tf.sparse.SparseTensor(
        indices=indices, values=values, dense_shape=data.shape
    )
    sparse_tensor = tf.sparse.reorder(sparse_tensor)
    get_id = lambda x: tf.gather(o_ids, x.indices)

    return (
        tf.data.Dataset.from_tensor_slices(sparse_tensor)
        .map(get_id, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


def asv_encode_and_convert(data):
    asv_dataset = convert_to_tf_dataset(data)
    sequence_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=5,
        split="character",
        output_mode="int",
        output_sequence_length=150,
    )
    sequence_tokenizer.adapt(asv_dataset.take(1))
    asv_dataset = asv_dataset.map(lambda x: sequence_tokenizer(x))

    return asv_dataset


def gotu_encode_and_convert(data):
    gotu_dataset = convert_to_tf_dataset(data)
    o_ids = tf.data.Dataset.from_tensor_slices(data.ids(axis="observation"))
    sequence_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=o_ids.cardinality().numpy()
        + 1,  # len of gotus + 1 for padding
        output_mode="int",
        output_sequence_length=1,
    )

    sequence_tokenizer.adapt(o_ids)
    gotu_list = data.ids(axis="observation")
    gotu_tokens = sequence_tokenizer(gotu_list) + 2

    gotu_dict = {
        int(t): gotu
        for t, gotu in zip(
            list(tf.squeeze(gotu_tokens, axis=-1).numpy()), gotu_list
        )
    }

    gotu_dataset = gotu_dataset.map(
        lambda x: (
            tf.concat(
                [
                    tf.constant([[1]], dtype=tf.int64),
                    sequence_tokenizer(x) + 2,
                    tf.constant([[2]], dtype=tf.int64),
                ],
                axis=0,
            )
        )
    )
    return gotu_dataset, gotu_dict


def combine_all_datasets(asv_dataset, gotu_dataset):
    return tf.data.Dataset.zip(asv_dataset, gotu_dataset).prefetch(
        tf.data.AUTOTUNE
    )


def batch_dataset(dataset, batch_size):
    def extract_zip(asv_data, gotu_data):
        return (
            {
                "encoder_inputs": asv_data,
                "decoder_inputs": gotu_data[:-1, :],
            },
            tf.squeeze(gotu_data[1:, :], axis=-1),
        )

    dataset = dataset.cache()
    size = dataset.cardinality()
    dataset = (
        dataset.map(extract_zip)
        .padded_batch(
            batch_size,
            padded_shapes=(
                {"encoder_inputs": [None, 150], "decoder_inputs": [None, 1]},
                [None],
            ),
            drop_remainder=True,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    dataset = dataset.cache()
    return dataset.prefetch(tf.data.AUTOTUNE)


def generate_train_val_sets(
    dataset: tf.data.Dataset, train_split: int, batch_size: int, shuffle: int
):
    size = dataset.cardinality().numpy()
    best_overlap = 0
    best_training_set = None
    best_val_set = None

    for i in range(shuffle):
        dataset = dataset.shuffle(size, reshuffle_each_iteration=True)
        dataset = dataset.cache()
        train_size = int(size * train_split / batch_size) * batch_size
        training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)

        for _, x in training_dataset:
            train_set = x.numpy()
        for _, y in val_dataset:
            val_set = y.numpy()

        set_overlap = len(np.intersect1d(train_set, val_set)) / len(val_set)
        print(f"Set Overlap: {set_overlap * 100}%")
        if set_overlap > best_overlap:
            best_overlap = set_overlap
            best_training_set = training_dataset
            best_val_set = val_dataset

    batched_training_dataset = batch_dataset(best_training_set, batch_size)
    batched_validation_dataset = batch_dataset(best_val_set, batch_size)
    print(f"Best Overlap: {best_overlap * 100}%")
    return batched_training_dataset, batched_validation_dataset


def create_training_dataset(gotu_fp: str, asv_fp: str, train_split: int):
    gotu_encoded, gotu_dict = gotu_encode_and_convert(load_biom_table(gotu_fp))
    asv_encoded = asv_encode_and_convert(load_biom_table(asv_fp))
    combined_dataset = combine_all_datasets(asv_encoded, gotu_encoded)
    training_batched, val_batched = generate_train_val_sets(
        combined_dataset, train_split, 8, 100
    )

    return training_batched, val_batched, gotu_dict


def create_prediction_dataset(gotu_fp, asv_fp, batch_size):
    gotu_encoded, gotu_dict = gotu_encode_and_convert(load_biom_table(gotu_fp))
    asv_encoded = asv_encode_and_convert(load_biom_table(asv_fp))
    combined_dataset = batch_dataset(
        combine_all_datasets(asv_encoded, gotu_encoded), batch_size
    )

    return combined_dataset, gotu_dict
