import tensorflow as tf


def cos_decay_with_warmup():
    decay_steps = 10000.0
    initial_learning_rate = 0.0
    warmup_steps = 10000.0
    target_learning_rate = 1e-4
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate,
        decay_steps,
        alpha=1e-5,
        warmup_target=target_learning_rate,
        warmup_steps=warmup_steps,
    )
    return lr
