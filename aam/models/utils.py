import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="TransformerLearningRateSchedule")
class TransformerLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    def __init__(
        self, d_model, warmup_steps=100, decay_method="cosine", initial_lr=3e-4
    ):
        super(TransformerLearningRateSchedule, self).__init__()

        self.d_model = d_model
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
            cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.initial_lr,
                first_decay_steps=1000,  # Change according to your training steps
                t_mul=2.0,  # How quickly to increase the restart periods
                m_mul=0.9,  # Multiplier for reducing max learning rate after each restart
                alpha=0.0,  # Minimum learning rate
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
                "d_model": self.d_model,
                "warmup_steps": self.warmup_steps,
                "decay_method": self.decay_method,
                "initial_lr": self.initial_lr,
            }
        )
        return config


def cos_decay_with_warmup():
    lr = tf.keras.optimizers.schedules.ExponentialDecay(3e-4, 1000, 0.96)
    return lr
