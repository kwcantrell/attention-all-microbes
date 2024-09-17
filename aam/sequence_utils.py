import tensorflow as tf

from aam.utils import float_mask


def add_random_sequences(seq, seeds, range):
    seq_mask = tf.cast(tf.math.equal(seq, 0), dtype=tf.int32)

    nucleotides = tf.reduce_sum(
        tf.one_hot(seq, depth=6, axis=-1, dtype=tf.float32), axis=[0, 1]
    )
    unormalized_log_prob = tf.divide(
        nucleotides, tf.math.reduce_max(nucleotides, axis=-1, keepdims=True)
    )
    unormalized_log_prob = tf.math.log(
        tf.math.divide_no_nan(
            unormalized_log_prob,
            tf.ones_like(unormalized_log_prob) - unormalized_log_prob,
        ),
    )
    nuc_strings = tf.multiply(
        tf.reshape(
            tf.transpose(
                tf.random.stateless_categorical(
                    unormalized_log_prob,
                    tf.cast(range, dtype=tf.int32),
                    tf.cast(seeds[:, 1], dtype=tf.int32),
                    dtype=tf.int32,
                )
            ),
            tf.shape(seq),
        ),
        seq_mask,
    )

    seq = tf.concat([seq, nuc_strings[:, -8:, :]], axis=1)
    return seq


def random_sequences_mask(seq, seeds, seq_mask_rate):
    random_mask = tf.random.stateless_uniform(tf.shape(seq), seeds[:, 0])
    random_mask = tf.cast(
        tf.math.greater_equal(random_mask, seq_mask_rate), dtype=tf.int32
    )
    seq = tf.multiply(seq, random_mask)
    return seq


def add_random_seq_and_mask(seq, seeds, range):
    seq = add_random_sequences(seq, seeds, range)
    return random_sequences_mask(seq, seeds, range)


def compute_pca_proj(tensor, hidden_dim, num_heads, head_size):
    BP_DIM = 2
    HEAD_DIM = 3
    HEAD_SIZE_DIM = 4
    shape = tf.shape(tensor)
    reshape = tf.concat([shape[:-1], [num_heads, head_size]], axis=0)
    perm = tf.concat(
        [tf.range(tf.shape(reshape)[0])[:-3], [HEAD_DIM, BP_DIM, HEAD_SIZE_DIM]], axis=0
    )
    second_reshape = tf.concat([shape[:-2], [hidden_dim, head_size]], axis=0)

    tensor = tf.subtract(tensor, tf.reduce_mean(tensor, axis=-2, keepdims=True))
    tensor = tf.reshape(tensor, shape=reshape)
    tensor = tf.transpose(tensor, perm=perm)
    cov = tf.linalg.matmul(tensor, tensor, transpose_a=True)
    eig_values, eig_vec = tf.linalg.eigh(cov)
    pca = tf.transpose(
        tf.matmul(
            tf.linalg.diag(eig_values),
            eig_vec,
        ),
        perm=perm,
    )
    pca = tf.reshape(pca, shape=second_reshape)
    return pca


def add_seq_and_count_pad(inputs):
    seq, rclr = tf.nest.flatten(inputs, expand_composites=True)
    seq = tf.pad(seq, [[0, 0], [0, 1], [0, 0]], constant_values=7)
    rclr = tf.pad(rclr, paddings=[[0, 0], [0, 1]], constant_values=1)
    return (seq, rclr)


def sequence_attention_mask(tensor):
    mask = float_mask(tensor)
    attention_mask = tf.cast(tf.matmul(mask, mask, transpose_b=True), dtype=tf.bool)
    return attention_mask
