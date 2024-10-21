import tensorflow as tf
from biom import load_table
from unifrac import unweighted


def load_data(
    table_path,
    shuffle_samples=True,
    tree_path=None,
    max_token_per_sample=512,
    batch_size=8,
    temp_table_path="temp_table.biom",
):
    def _get_unifrac_data(table_path, tree_path):
        distance = unweighted(table_path, tree_path).data
        return tf.data.Dataset.from_tensor_slices(distance)

    def _get_table_data(table_path):
        table_data = load_table(table_path)
        coo = table_data.transpose().matrix_data.tocoo()
        (data, (row, col)) = (coo.data, coo.coords)
        data = data
        row = row
        col = col
        return data, row, col

    def _preprocess_table(table_path):
        # table = load_table(table_path)
        # table = table.remove_empty()
        # table_data = table.matrix_data.tocoo()
        # counts, (row, col) = table_data.data, table_data.coords

        # for i in range(table.shape[1]):
        #     sample_mask = col == i
        #     sample_counts = counts[sample_mask]
        #     ranks = rankdata(sample_counts, method="ordinal")
        #     max_rank = np.max(ranks)
        #     rank_mask = ranks <= (max_rank - max_token_per_sample)
        #     sample_counts[rank_mask] = 0
        #     counts[sample_mask] = sample_counts
        # filtered_table_data = csr_matrix((counts, (row, col)), shape=table.shape)
        # new_table = Table(
        #     filtered_table_data, table.ids(axis="observation"), table.ids()
        # )
        # with biom_open(temp_table_path, "w") as f:
        #     new_table.to_hdf5(f, "aam")
        return "/home/kalen/aam-research-exam/research-exam/healty-age-regression/temp_table.biom"

    table_path = _preprocess_table(table_path)
    data, row, col = _get_table_data(table_path)
    unifrac_data = _get_unifrac_data(table_path, tree_path)

    table = load_table(table_path)
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
            def process_table(sample, data, unifrac_data):
                sorted_order = tf.argsort(data.values, axis=-1, direction="DESCENDING")

                asv_indices = tf.reshape(data.indices, shape=[-1])
                sorted_asv_indices = tf.gather(asv_indices, sorted_order)[
                    :max_token_per_sample
                ]

                encodings = tf.gather(asv_encodings, sorted_asv_indices)
                tokens = lookup_table.lookup(encodings).to_tensor()
                return sample, tf.cast(tokens, dtype=tf.int32), unifrac_data

            def filter(samples, table_data, unifrac_data):
                return (
                    (table_data, table_data[:, :, :1]),
                    tf.gather(unifrac_data, samples, axis=1),
                )

            ds = ds.map(process_table, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.cache()
            if shuffle_samples and not val:
                ds = ds.shuffle(shuffle_buf)
            ds = ds.padded_batch(
                batch_size,
                drop_remainder=True,
            )
            ds = ds.map(filter, num_parallel_calls=tf.data.AUTOTUNE)

            return ds

        return _inner

    dataset_size = len(s_ids)
    training_size = int(dataset_size * 0.9)

    samples = tf.data.Dataset.from_tensor_slices(list(range(len(s_ids))))
    dataset = tf.data.Dataset.zip(samples, table, unifrac_data)
    train_dataset = (
        dataset.take(training_size)
        .apply(process_dataset(val=False, shuffle_buf=training_size))
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        dataset.skip(training_size)
        .apply(process_dataset(val=True))
        .prefetch(tf.data.AUTOTUNE)
    )
    data_obj = {
        "sample_ids": s_ids,
        "o_ids": o_ids,
        "training_dataset": train_dataset,
        "validation_dataset": val_dataset,
        "temp_table_file_path": table_path,
    }
    return data_obj
