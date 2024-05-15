import tensorflow as tf

from aam.common.losses import PairwiseLoss
from aam.common.metrics import PairwiseMAE


@tf.keras.saving.register_keras_serializable(package="UnifracModel")
class UnifracModel(tf.keras.Model):
    def __init__(self, max_bp, batch_size, feature_emb, readhead, **kwargs):
        super().__init__(**kwargs)
        self.max_bp = max_bp
        self.batch_size = batch_size
        self.feature_emb = feature_emb
        self.readhead = readhead
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pair_loss = PairwiseLoss()
        self.metric_traker = PairwiseMAE()
        self.make_call_function()

    def make_call_function(self):
        @tf.autograph.experimental.do_not_convert
        def one_step(inputs, training=None):
            ind, seq = inputs
            output = self.feature_emb(seq, training=training)
            output = self.readhead(output)
            return output

        self.call_function = tf.function(one_step, reduce_retracing=True)

    def call(self, inputs, training=None):
        return self.call_function(inputs, training=training)

    def build(self, input_shape):
        input = tf.keras.Input(
            shape=[None, self.max_bp], batch_size=self.batch_size, dtype=tf.int64
        )
        output = self.feature_emb(input)
        output = self.readhead(output)

    def train_step(self, data):
        x, y = data
        ind, seq = tf.nest.flatten(x)
        ind = tf.squeeze(ind)
        y = tf.gather(y, ind, axis=1, batch_dims=0)

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self((ind, seq), training=True)

            # Compute regression loss
            loss = self.pair_loss(y, outputs)
            self.loss_tracker.update_state(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.metric_traker.update_state(y, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.metric_traker.result(),
        }

    def test_step(self, data):
        x, y = data
        ind, seq = tf.nest.flatten(x)
        ind = tf.squeeze(ind)
        y = tf.gather(y, ind, axis=1, batch_dims=0)
        # Forward pass
        outputs = self((ind, seq), training=False)

        # Compute regression loss
        loss = self.pair_loss(y, outputs)
        self.loss_tracker.update_state(loss)

        # Compute our own metrics
        self.metric_traker.update_state(y, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.metric_traker.result(),
        }

    def get_config(self):
        base_config = super().get_config()
        config = {
            "max_bp": self.max_bp,
            "batch_size": self.batch_size,
            "feature_emb": tf.keras.saving.serialize_keras_object(self.feature_emb),
            "readhead": tf.keras.saving.serialize_keras_object(self.readhead),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["feature_emb"] = tf.keras.saving.deserialize_keras_object(
            config["feature_emb"]
        )
        config["readhead"] = tf.keras.saving.deserialize_keras_object(
            config["readhead"]
        )
        return cls(**config)
