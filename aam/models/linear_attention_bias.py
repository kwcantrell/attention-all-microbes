import tensorflow as tf


def _construct_bias(inputs):
    shape = tf.shape(inputs)
    query_len = shape[2]
    key_len = shape[3]
    num_heads = shape[1]

    largest_len = tf.where(query_len > key_len, query_len, key_len)

    bias = tf.repeat(
        tf.expand_dims(tf.range(0, largest_len, 1, dtype=tf.float32), axis=0),
        repeats=largest_len,
        axis=0,
    )
    bias_mask = tf.cast(
        tf.expand_dims(tf.range(0, largest_len, 1, dtype=tf.float32), axis=-1) >= bias,
        dtype=tf.float32,
    )
    bias = -1 * tf.sort(bias * bias_mask, direction="DESCENDING")
    bias = tf.expand_dims(bias, axis=0)
    bias = tf.expand_dims(bias, axis=0)

    start = tf.math.log(tf.cast(num_heads, dtype=tf.float32)) / tf.math.log(2.0)
    start = 2 ** (-(2 ** -(start - 3)))
    m = tf.map_fn(
        lambda i: start * tf.pow(start, i),
        tf.range(num_heads, dtype=tf.float32),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    m = tf.expand_dims(m, axis=0)
    m = tf.expand_dims(m, axis=-1)
    m = tf.expand_dims(m, axis=-1)
    alibi = bias * m
    alibi = tf.cast(
        alibi + tf.transpose(alibi, perm=[0, 1, 3, 2]),
        dtype=tf.keras.mixed_precision.global_policy().compute_dtype,
    )
    alibi = alibi[:, :, :query_len, :key_len]
    tf.print("ALIBI: ", alibi)
    return alibi


def _large_compatible_negative(tensor_type):
    """Large negative number as Tensor.

    This function is necessary because the standard value for epsilon
    in this module (-1e9) cannot be represented using tf.float16

    Args:
        tensor_type: a dtype to determine the type.

    Returns:
        a large negative number.
    """
    # In case of dtype=float16 (e.g., for mixed-precision), the largest
    # negative number (dtypes.float16.min) is divided by 2, in order to
    # avoid overflows when summing negative inputs.
    if tensor_type == tf.float16:
        return tf.float16.min / 2.0
    return -1e9


@tf.keras.saving.register_keras_serializable(package="LinearBiasSoftmax")
class LinearBiasSoftmax(tf.keras.layers.Layer):
    """LinearBiasSoftmax activation function.

    Example without mask:

    >>> inp = np.asarray([[1., 2., 1.]])
    >>> layer = tf.keras.layers.LinearBiasSoftmax()
    >>> layer(inp).numpy()
    array([[0.21194157, 0.5761169 , 0.21194157]], dtype=float32)
    >>> mask = np.asarray([[True, False, True]], dtype=bool)
    >>> layer(inp, mask).numpy()
    array([[0.5, 0. , 0.5]], dtype=float32)

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        axis: Integer, or list of Integers, axis along which the softmax
            normalization is applied.
    Call arguments:
        inputs: The inputs, or logits to the softmax layer.
        mask: A boolean mask of the same shape as `inputs`. The mask
            specifies 1 to keep and 0 to mask. Defaults to `None`.


    Returns:
        Softmaxed output with the same shape as `inputs`.
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        shape = [s if s is not None else 1 for s in input_shape]
        t = tf.ones(shape)
        bias = _construct_bias(t)
        self.bias = lambda: bias

    def call(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            adder = (1.0 - tf.cast(mask, inputs.dtype)) * (
                _large_compatible_negative(inputs.dtype)
            )

            # Since we are adding it to the raw scores before the softmax, this
            # is effectively the same as removing these entirely.
            inputs += adder
        inputs += self.bias()
        return tf.keras.backend.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
