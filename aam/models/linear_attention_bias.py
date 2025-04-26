import tensorflow as tf


def _construct_bias(input_shape, sequence=None):
    num_heads = input_shape[1]
    query_len = input_shape[2]
    key_len = input_shape[3]

    largest_len = tf.reduce_max([query_len, key_len])
    if sequence is None:
        sequence = tf.range(0, largest_len, 1, dtype=tf.float32)
    else:
        sequence = tf.cast(tf.squeeze(sequence), dtype=tf.float32)

    bias = -1 * tf.abs(tf.expand_dims(sequence, axis=-1) - tf.expand_dims(sequence, 0))
    bias = tf.expand_dims(bias, axis=0)
    bias = tf.expand_dims(bias, axis=0)
    bias = bias[:, :, :query_len, :key_len]
    m = tf.map_fn(
        lambda i: tf.pow(tf.constant(2, dtype=tf.float32), -(i + 1)),
        tf.range(num_heads, dtype=tf.float32),
    )

    m = tf.expand_dims(m, axis=0)
    m = tf.expand_dims(m, axis=-1)
    m = tf.expand_dims(m, axis=-1)
    return m, bias


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

    def __init__(self, fix_bias_shape, use_sparse_positions, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.fix_bias_shape = fix_bias_shape
        self.use_sparse_positions = use_sparse_positions
        if self.use_sparse_positions:
            print("using sparse positions")

    # def build(self, input_shape):
    #     if self.fix_bias_shape:
    #         print("fixing bias shape:", input_shape)
    #         input_shape = tf.TensorShape(input_shape)
    #         self.m, self.bias = _construct_bias(input_shape)

    def construct_bias(self, inputs, mask):
        if not self.fix_bias_shape:
            if self.use_sparse_positions:
                m, bias = _construct_bias(tf.shape(inputs), mask)
            else:
                m, bias = _construct_bias(tf.shape(inputs))
            alibi = m * bias
        else:
            alibi = self.m * self.bias
        return tf.cast(alibi, dtype=self.compute_dtype)

    def call(self, inputs, mask=None):
        # alibi = self.construct_bias(inputs, mask)
        input_shape = tf.shape(inputs)
        num_heads = input_shape[1]
        query_len = input_shape[2]
        key_len = input_shape[3]

        largest_len = tf.reduce_max([query_len, key_len])
        sequence = tf.range(0, largest_len, 1, dtype=tf.float32)
        # else:
        #     sequence = tf.cast(tf.squeeze(sequence), dtype=tf.float32)

        bias = -1 * tf.abs(tf.expand_dims(sequence, axis=-1) - tf.expand_dims(sequence, 0))
        bias = tf.expand_dims(bias, axis=0)
        bias = tf.expand_dims(bias, axis=0)
        bias = bias[:, :, :query_len, :key_len]
        m = tf.map_fn(
            lambda i: tf.pow(tf.constant(2, dtype=tf.float32), -(i + 1)),
            tf.range(num_heads, dtype=tf.float32),
        )

        m = tf.expand_dims(m, axis=0)
        m = tf.expand_dims(m, axis=-1)
        m = tf.expand_dims(m, axis=-1)
        alibi = m * bias
        alibi = tf.cast(alibi, dtype=self.compute_dtype)
        tf.print("mask", tf.shape(mask))
        if not self.use_sparse_positions and mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            adder = (1.0 - tf.cast(mask, inputs.dtype)) * (_large_compatible_negative(inputs.dtype))

            # Since we are adding it to the raw scores before the softmax, this
            # is effectively the same as removing these entirely.
            inputs += adder
        inputs += alibi
        return tf.keras.backend.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {
            "fix_bias_shape": self.fix_bias_shape,
            "use_sparse_positions": self.use_sparse_positions,
            "axis": self.axis,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
