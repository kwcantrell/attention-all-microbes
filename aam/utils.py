import functools

import tensorflow as tf


def float_mask(tensor: tf.Tensor, dtype=tf.float32) -> tf.Tensor:
    """creates a mask for nonzero elements of tensor. I.e. mask*tensor = tensor and
    (1. - mask) * tensor = 0

    Args:
        tensor (tf.Tensor): a tensor of type float

    Returns:
        tf.Tensor:
    """
    mask = tf.cast(tf.not_equal(tensor, 0), dtype=dtype)
    return mask


def create_random_mask(
    shape: tf.Tensor, percent: tf.Tensor, dtype: tf.DType = tf.float32
) -> tf.Tensor:
    random_mask = tf.random.uniform(
        shape,
        maxval=1,
        dtype=tf.keras.mixed_precision.global_policy().compute_dtype,
    )
    random_mask = tf.cast(
        random_mask <= percent,
        dtype=dtype,
    )
    return random_mask


def apply_random_mask(tensor: tf.Tensor, mask_percent: float) -> tf.Tensor:
    if not isinstance(mask_percent, float):
        raise Exception("Invalid mask percent")

    dtype = tf.type_spec_from_value(tensor).dtype
    mask = float_mask(tensor)
    random_mask = tf.random.uniform(tf.shape(tensor), minval=0, maxval=1)
    random_mask = tf.cast(
        tf.greater_equal(random_mask, mask_percent),
        dtype=tf.keras.mixed_precision.global_policy().compute_dtype,
    )
    mask = mask * random_mask
    return tensor * tf.cast(mask, dtype=dtype)


def masked_loss(sparse_cat: bool = False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(obj, target, pred, **kwargs):
            shape = tf.shape(pred)

            if sparse_cat:
                target = tf.reshape(target, [shape[0], -1])
                mask = float_mask(target)
                pred = tf.reshape(pred, [shape[0], -1, shape[-1]])
            else:
                target = tf.reshape(target, [shape[0], -1, 1])
                mask = float_mask(target)
                mask = tf.squeeze(mask, axis=-1)
                pred = tf.reshape(pred, [shape[0], -1, 1])

            loss = func(obj, target, pred, **kwargs)

            total = tf.cast(tf.reduce_sum(mask), dtype=tf.float32)
            loss = tf.reduce_sum(loss * mask)
            # return tf.math.divide_no_nan(loss, total)
            return loss

        return wrapper

    return decorator


def load_model(fp):
    """Important! tf.keras.models.load_model does not properly restore weights.
    This function will properly restore weights from a .keras file
    """
    import importlib
    import json
    from zipfile import ZipFile

    with ZipFile(fp, "r") as zobj:
        zobj.extractall(".")
        config = json.loads(zobj.read("config.json").decode("utf-8"))
        build_config = config["build_config"]
        model_config = config["config"]
        if "build_input_shape" in model_config:
            model_config.pop("build_input_shape")
        module = importlib.import_module(config["module"])
        cls = getattr(module, config["class_name"])
        model = cls(**model_config)
        model.build(build_config["input_shape"])
        model.load_weights("model.weights.h5")
    return model
