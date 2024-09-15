import tensorflow as tf
from aam.layers import TransferAttention


@tf.keras.saving.register_keras_serializable(package="TransferLearnNucleotideModel")
class TransferLearnNucleotideModel(tf.keras.Model):
    def __init__(
        self,
        base_model,
        **kwargs,
    ):
        super(TransferLearnNucleotideModel, self).__init__(**kwargs)

        self.regresssion_loss = tf.keras.losses.MeanSquaredError()
        self.loss_tracker = tf.keras.metrics.Mean()
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError()

        # layers used in model
        self.base_model = base_model
        self.base_model.trainable = False

        self.attention_layer = TransferAttention(
            0, hidden_dim=128, name="transfer_output"
        )
        self.linear_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)

        def _compute_loss(y_true, y_pred):
            loss = self.regresssion_loss(y_true, y_pred)
            return loss

        self._compute_loss = _compute_loss

    def build(self, input_shape=None):
        input_seq = tf.keras.Input(
            shape=[None, 150],
            dtype=tf.int32,
        )
        input_rclr = tf.keras.Input(
            shape=[None],
            dtype=tf.float32,
        )
        outputs = self.call((input_seq, input_rclr), training=False)
        self.inputs = (input_seq, input_rclr)
        self.outputs = outputs
        super(TransferLearnNucleotideModel, self).build(
            (tf.TensorShape([None, None, 150]), tf.TensorShape([None, None]))
        )

    def predict_step(self, data):
        inputs, y = data
        y_pred = self(inputs, training=False)
        return tf.squeeze(y_pred), tf.squeeze(y)

    def train_step(self, data):
        inputs, y = data
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self._compute_loss(y, y_pred)
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_tracker.result()}

    def test_step(self, data):
        inputs, y = data

        y_pred = self(inputs, training=False)
        loss = self._compute_loss(y, y_pred)

        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_tracker.result()}

    def call(self, inputs, training=False):
        base_outputs = self.base_model.feature_emb(
            inputs, return_nuc_attention=False, training=False
        )
        reg_out = self.attention_layer(base_outputs, training=training)
        reg_out = self.linear_activation(reg_out)
        return reg_out

    def get_config(self):
        config = super(TransferLearnNucleotideModel, self).get_config()
        config.update(
            {
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
            }
        )
        return config
