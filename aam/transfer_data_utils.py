import numpy as np
import scipy.stats
import pandas as pd
import tensorflow as tf
from biom import load_table


def validate_metadata(table, metadata, missing_samples_flag):
    # check for mismatch samples
    ids = table.ids(axis="sample")
    shared_ids = set(ids).intersection(set(metadata.index))
    min_ids = min(len(shared_ids), len(ids), len(metadata.index))
    max_ids = max(len(shared_ids), len(ids), len(metadata.index))
    if len(shared_ids) == 0:
        raise Exception("Table and Metadata have no matching sample ids")
    if min_ids != max_ids and missing_samples_flag == "error":
        raise Exception("Table and Metadata do not share all same sample ids.")
    elif min_ids != max_ids and missing_samples_flag == "ignore":
        print("Warning: Table and Metadata do not share all same sample ids.")
        print("Table and metadata will be filtered")
        table = table.filter(shared_ids, inplace=False)
        metadata = metadata[metadata.index.isin(shared_ids)]
    return table, metadata


def filter_and_reorder(metadata, ids):
    metadata = metadata[metadata.index.isin(ids)]
    metadata = metadata.reindex(ids)
    return metadata


def extract_col(metadata, col, is_categorical):
    metadata_col = metadata[col]
    if not is_categorical:
        metadata_col = metadata_col.astype(np.float32)
        return metadata_col

    metadata_col = metadata_col.astype("category")
    return metadata_col


def load_data(
    table_path,
    is_categorical,
    metadata_path,
    metadata_col,
    class_labels=None,
    shuffle_samples=True,
    missing_samples_flag=None,
    max_token_per_sample=225,
):
    def _get_table_data(table_data):
        coo = table_data.transpose().matrix_data.tocoo()
        (data, (row, col)) = (coo.data, coo.coords)
        data = data
        row = row
        col = col
        return data, row, col

    table = load_table(table_path)
    shuffle_ids = table.ids()
    np.random.shuffle(shuffle_ids)
    table = table.sort_order(shuffle_ids, axis="sample")

    df = pd.read_csv(metadata_path, sep="\t", index_col=0)[[metadata_col]]
    cat_labels = None
    table, df = validate_metadata(table, df, missing_samples_flag)
    filter_ids = table.ids()
    df = filter_and_reorder(df, filter_ids)
    target_data = extract_col(df, metadata_col, is_categorical=is_categorical)
    cat_labels = None
    cat_counts = None
    num_classes = None
    max_density = None
    mean = None
    std = None
    if is_categorical:
        cat_labels = target_data.cat.categories
        cat_counts = target_data.value_counts()
        cat_counts = cat_counts.reindex(cat_labels).to_numpy().astype(np.float32)
        target_data = target_data.cat.codes
        num_classes = np.max(target_data) + 1
        target_data = tf.expand_dims(target_data, axis=-1)
        target_dataset = tf.data.Dataset.from_tensor_slices(target_data)
    else:
        mean = np.mean(target_data)
        std = np.std(target_data)
        target_data = (target_data - mean) / std
        density = scipy.stats.gaussian_kde(target_data)
        y = density(target_data).astype(np.float32)
        max_density = np.max(y)
        target_data = tf.expand_dims(target_data, axis=-1)
        y = tf.expand_dims(y, axis=-1)
        target_dataset = tf.data.Dataset.from_tensor_slices(target_data)
        y_dataset = tf.data.Dataset.from_tensor_slices(y)
        target_dataset = tf.data.Dataset.zip(target_dataset, y_dataset)

    s_ids = table.ids()
    o_ids = table.ids(axis="observation")

    data, row, col = _get_table_data(table)

    s_ids = table.ids()
    o_ids = table.ids(axis="observation")
    indices = tf.concat(
        [tf.expand_dims(row, axis=1), tf.expand_dims(col, axis=1)], axis=1
    )
    table = tf.sparse.SparseTensor(
        indices=tf.cast(indices, dtype=tf.int64),
        values=data,
        dense_shape=[len(s_ids), len(o_ids)],
    )
    table = tf.sparse.reorder(table)
    table = tf.data.Dataset.from_tensor_slices(table)

    # These are the UTF-8 encodings of A, C, T, G respectively
    # lookup table converts utf-8 encodings to token
    # tokens start at 1 to make room for pad token
    key_val_init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([65, 67, 71, 84], dtype=tf.int64),
        values=tf.constant([1, 2, 3, 4], dtype=tf.int64),
    )
    lookup_table = tf.lookup.StaticVocabularyTable(key_val_init, num_oov_buckets=1)
    asv_encodings = tf.cast(tf.strings.unicode_decode(o_ids, "UTF-8"), dtype=tf.int64)

    def process_dataset(val=False, shuffle_buf=100):
        def _inner(ds):
            def process_table(data, target_data):
                sorted_order = tf.argsort(data.values, axis=-1, direction="DESCENDING")

                asv_indices = tf.reshape(data.indices, shape=[-1])
                sorted_asv_indices = tf.gather(asv_indices, sorted_order)[
                    :max_token_per_sample
                ]
                counts = tf.gather(data.values, sorted_order)[:max_token_per_sample]
                counts = tf.math.log1p(counts)

                encodings = tf.gather(asv_encodings, sorted_asv_indices)
                tokens = lookup_table.lookup(encodings).to_tensor()
                if is_categorical:
                    return (tf.cast(tokens, dtype=tf.int32), counts), tf.squeeze(
                        target_data, axis=-1
                    )
                else:
                    return (tf.cast(tokens, dtype=tf.int32), counts), target_data

            def filter(table_data, target_data):
                return table_data, target_data

            if shuffle_samples and not val:
                ds = ds.shuffle(shuffle_buf)
            ds = ds.map(process_table, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.padded_batch(8)
            ds = ds.map(filter, num_parallel_calls=tf.data.AUTOTUNE)

            return ds

        return _inner

    dataset_size = len(s_ids)
    training_size = int(dataset_size * 0.9)

    dataset = tf.data.Dataset.zip(table, target_dataset)
    train_dataset = (
        dataset.take(training_size)
        .cache()
        .apply(process_dataset(val=False, shuffle_buf=training_size))
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        dataset.skip(training_size)
        .cache()
        .apply(process_dataset(val=True))
        .prefetch(tf.data.AUTOTUNE)
    )
    data_obj = {
        "sample_ids": s_ids,
        "o_ids": o_ids,
        "training_dataset": train_dataset,
        "validation_dataset": val_dataset,
        "metadata": df,
        "num_classes": num_classes,
        "cat_labels": cat_labels,
        "cat_counts": cat_counts,
        "max_density": max_density,
        "mean": mean,
        "std": std,
    }
    return data_obj
