import tensorflow as tf


class BaseLoss(tf.keras.losses.Loss):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def call(self, y_true, y_pred):
        return tf.reduce_mean(
            self.loss_fn(y_true, y_pred),
            axis=-1
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "loss_fn": self.loss_fn
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(
    package="FeaturePresent"
)
class FeaturePresent(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = tf.keras.losses.sparse_categorical_crossentropy

    def call(self, mask, y_pred):

        mask = tf.cast(mask, dtype=tf.int64)
        return tf.reduce_mean(
            self.loss(mask, y_pred, from_logits=True),
            axis=-1
        )

    def get_input(self, model_outputs, dataset_outputs=None):
        token_mask = model_outputs["token_mask"]
        embeddings = model_outputs["embeddings"]
        return [
            token_mask,
            embeddings
        ]

    def get_config(self):
        return super().get_config()
