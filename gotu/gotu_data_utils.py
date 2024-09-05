"""gotu_data_utils.py"""

import json

import numpy as np
import tensorflow as tf
from biom import load_table


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
    with open(full_path, "w") as file:
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
    with open(file_path, "r") as json_file:
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


def gotu_encode_and_convert(data):
    gotu_dataset = convert_to_tf_dataset(data)
    o_ids = tf.data.Dataset.from_tensor_slices(data.ids(axis="observation"))
    gotu_count = o_ids.cardinality().numpy()
    sequence_tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=gotu_count + 1,  # len of gotus + 1 for padding
        output_mode="int",
        output_sequence_length=1,
    )

    sequence_tokenizer.adapt(o_ids)
    gotu_list = data.ids(axis="observation")
    gotu_tokens = sequence_tokenizer(gotu_list) + 2

    gotu_dict = {
        int(t): gotu
        for t, gotu in zip(list(tf.squeeze(gotu_tokens, axis=-1).numpy()), gotu_list)
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
    return gotu_dataset, gotu_dict, gotu_count
