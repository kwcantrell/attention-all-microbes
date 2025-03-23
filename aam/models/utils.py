import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="TransformerLearningRateSchedule")
class TransformerLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    def __init__(self, warmup_steps=100, decay_method="cosine", initial_lr=3e-4):
        super(TransformerLearningRateSchedule, self).__init__()

        self.warmup_steps = warmup_steps
        self.decay_method = decay_method
        self.initial_lr = initial_lr

    def __call__(self, step):
        # Linear warmup
        learning_rate = tf.cast(
            self.initial_lr * tf.math.minimum(step / self.warmup_steps, 1.0),
            dtype=tf.float32,
        )

        if self.decay_method == "cosine":
            # Cosine decay after warmup
            cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.initial_lr,
                first_decay_steps=self.warmup_steps,
            )
            learning_rate = tf.cond(
                step < self.warmup_steps,
                lambda: learning_rate,
                lambda: cosine_decay(step - self.warmup_steps),
            )
        elif self.decay_method == "inv_sqrt":
            # Inverse Square Root decay after warmup (used in the original Transformer paper)
            inv_sqrt_decay = self.initial_lr * tf.math.rsqrt(
                tf.cast(step - self.warmup_steps + 1, tf.float32)
            )
            learning_rate = tf.cond(
                step < self.warmup_steps, lambda: learning_rate, lambda: inv_sqrt_decay
            )

        return learning_rate

    def get_config(self):
        config = {}
        config.update(
            {
                "warmup_steps": self.warmup_steps,
                "decay_method": self.decay_method,
                "initial_lr": self.initial_lr,
            }
        )
        return config


def cos_decay_with_warmup(lr, warmup_steps=5000, decay_steps=1000):
    # # Learning rate schedule: Warmup followed by cosine decay
    # lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    #     initial_learning_rate=lr, first_decay_steps=warmup_steps
    # )
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=decay_steps,
        warmup_target=lr,
        warmup_steps=warmup_steps,
    )
    return lr_schedule


def to_batch(tensor, batch_counts):
    batch_size = tf.shape(batch_counts)[0]
    tensor_shape = tf.shape(tensor)[1:]

    batch_indices = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=-1)

    batch_counts = tf.cast(batch_counts, dtype=tf.int32)
    max_count = tf.reduce_max(batch_counts)
    batch_indices = tf.broadcast_to(batch_indices, shape=[batch_size, max_count])
    seq_indices = tf.tile(
        tf.expand_dims(tf.range(max_count, dtype=tf.int32), axis=0),
        multiples=[batch_size, 1],
    )

    batch_mask = seq_indices < tf.expand_dims(batch_counts, axis=-1)
    batch_indices = tf.stack([batch_indices, seq_indices], axis=-1)

    batch_mask = tf.reshape(batch_mask, shape=[-1])
    batch_indices = tf.reshape(batch_indices, shape=[-1, 2])[batch_mask]

    output_shape = tf.concat([[batch_size], [max_count], tensor_shape], axis=0)
    return tf.scatter_nd(batch_indices, updates=tensor, shape=output_shape)


def sort_using_counts(tensor, counts):
    sorted_indices = tf.argsort(
        tf.squeeze(counts, axis=-1), axis=1, direction="DESCENDING"
    )
    sorted_tensor = tf.gather(tensor, sorted_indices, axis=1, batch_dims=1)
    sorted_counts = tf.gather(counts, sorted_indices, axis=1, batch_dims=1)
    return sorted_tensor, sorted_counts


def batch_embeddings(asv_embeddings, batch_indicies, counts, asv_indices):
    emb_dim = tf.shape(asv_embeddings)[-1]
    batch_indicies = tf.cast(batch_indicies, dtype=tf.int32)
    asv_indices = tf.cast(asv_indices, dtype=tf.int32)

    if asv_indices is not None:
        asv_embeddings = tf.gather(asv_embeddings, asv_indices)
    batch_shape = tf.reduce_max(batch_indicies[:, 0]) + 1
    max_unique = tf.reduce_max(batch_indicies[:, 1]) + 1
    batch_embeddings = tf.scatter_nd(
        batch_indicies, asv_embeddings, shape=[batch_shape, max_unique, emb_dim]
    )
    counts = tf.scatter_nd(batch_indicies, counts, shape=[batch_shape, max_unique, 1])
    return batch_embeddings, counts


def sample_embeddings(asv_embeddings, batch_indicies, counts, asv_indices):
    batched_embeddigns, batch_counts = batch_embeddings(
        asv_embeddings, batch_indicies, counts, asv_indices
    )
    asv_mask = tf.cast(batch_counts > 0, dtype=tf.float32)
    batched_embeddigns = batched_embeddigns * asv_mask
    sample_embeddings = tf.reduce_sum(batched_embeddigns, axis=1) / tf.reduce_sum(
        asv_mask, axis=1
    )
    return sample_embeddings


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=4000,
        warmup_target=0.0003,
        warmup_steps=100,
    )

    # Generate learning rates for a range of steps
    steps = np.arange(4000, dtype=np.float32)
    learning_rates = [cosine_decay(step).numpy() for step in steps]

    # Plot the learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.plot(steps, learning_rates)
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Transformer Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.show()
