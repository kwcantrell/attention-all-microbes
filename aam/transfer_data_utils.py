import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def validate_metadata(table, metadata, missing_samples_flag):
    # check for mismatch samples
    ids = table.ids(axis="sample")
    # shared_ids = set(ids).intersection(set(metadata.index))
    shared_ids = np.intersect1d(ids, metadata.index)
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
        metadata = metadata.loc[shared_ids]
    return table.ids(), table, metadata


def shuffle(table, metadata):
    ids = table.ids()
    np.random.shuffle(ids)
    table = table.sort_order(ids, axis="sample")
    metadata = metadata.reindex(ids)
    return table, metadata


def extract_col(metadata, col, is_categorical):
    metadata_col = metadata[col]
    if not is_categorical:
        metadata_col = metadata_col.astype(np.float32)
        return metadata_col

    metadata_col = metadata_col.astype("category")
    return metadata_col


def load_data(
    table,
    is_categorical,
    df,
    metadata_col=None,
    class_labels=None,
    shuffle_samples=True,
    missing_samples_flag=None,
    max_token_per_sample=512,
    batch_size=8,
    shift=None,
    scale="minmax",
):
    from aam.data_handlers import SequenceTable

    st = SequenceTable(
        table,
        df,
        "/home/kalen/aam-research-exam/research-exam/healty-age-regression/taxonomy.tsv",
    )
    return st.get_data(
        max_token_per_sample,
        metadata_col,
        is_categorical,
        shift,
        scale,
        tax_level="Level 5",
        shuffle=shuffle_samples,
        batch_size=batch_size,
    )
