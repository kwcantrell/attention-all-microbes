import tensorflow as tf
import tensorflow_models as tfm

from aam.layers import CountEncoder
from aam.utils import float_mask


@tf.keras.saving.register_keras_serializable(package="TransferLearnNucleotideModel")
class TransferLearnNucleotideModel(tf.keras.Model):
    def __init__(
        self,
        base_model,
        freeze_base_weights,
        mask_percent=25,
        num_classes=None,
        shift=0,
        scale=1,
        penalty=5000,
        dropout=0.0,
        **kwargs,
    ):
        super(TransferLearnNucleotideModel, self).__init__(**kwargs)

        self.token_dim = 128
        self.mask_percent = mask_percent
        self.num_classes = num_classes
        self.shift = shift
        self.scale = scale
        self.penalty = tf.constant(penalty, dtype=tf.float32)
        self.dropout = dropout
        self.loss_tracker = tf.keras.metrics.Mean()
        self.target_tracker = tf.keras.metrics.Mean()
        self.reg_tracker = tf.keras.metrics.Mean()

        # layers used in model
        self.base_model = base_model
        self.freeze_base_weights = freeze_base_weights
        self.base_model.trainable = not self.freeze_base_weights

        # count embeddings

        self.count_encoder = CountEncoder(dropout_rate=self.dropout)
        self.count_intermediate = tf.keras.layers.Dense(
            128,
            activation="relu",
        )
        self.count_out = tf.keras.layers.Dense(1, use_bias=True)
        self.count_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        self.count_tracker = tf.keras.metrics.Mean()

        self.transfer_token = self.add_weight(
            "transfer_token",
            [1, 1, self.token_dim],
            dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )
        self.transfer_encoder = tfm.nlp.models.TransformerDecoder(
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            dropout_rate=self.dropout,
        )

        self.transfer_intermediate = tf.keras.layers.Dense(
            128,
            activation="relu",
        )
        if self.num_classes is None:
            self.transfer_ff = tf.keras.layers.Dense(1, use_bias=True)
            self.transfer_tracker = tf.keras.metrics.MeanAbsoluteError()
            self.transfer_string = "mae"
            self.transfer_activation = tf.keras.layers.Activation(
                "linear", dtype=tf.float32
            )
        else:
            self.transfer_ff = tf.keras.layers.Dense(num_classes)
            self.transfer_tracker = tf.keras.metrics.CategoricalAccuracy()
            self.transfer_string = "accuracy"
            self.transfer_activation = tf.keras.layers.Activation(
                "softmax", dtype=tf.float32
            )
        self.loss_metrics = sorted(
            ["loss", "target_loss", "count_mse", self.transfer_string]
        )

    def evaluate_metric(self, dataset, metric, **kwargs):
        metric_index = self.loss_metrics.index(metric)
        evaluated_metrics = super(TransferLearnNucleotideModel, self).evaluate(
            dataset, **kwargs
        )
        return evaluated_metrics[metric_index]

    def _compute_loss(self, y_true, outputs):
        _, count_pred, y_pred, counts = outputs

        y_pred = tf.ensure_shape(y_pred, [None, 1])
        count_pred = tf.ensure_shape(count_pred, [None, None])

        counts = tf.cast(counts, dtype=tf.float32)

        # count mask
        count_mask = float_mask(counts)
        count_mask = tf.ensure_shape(count_mask, [None, None])

        num_counts = tf.reduce_sum(count_mask, axis=-1, keepdims=True)
        num_counts = tf.ensure_shape(num_counts, [None, 1])

        # count mse
        count_loss = tf.math.square(counts - count_pred) * count_mask
<<<<<<< HEAD
        # count_loss = tf.math.divide_no_nan(count_loss, counts)
=======
        count_loss = tf.ensure_shape(count_loss, [None, None])
>>>>>>> checkpoint
        count_loss = tf.reduce_sum(count_loss, axis=-1, keepdims=True) / num_counts
        count_loss = tf.ensure_shape(count_loss, [None, 1])

        target_loss = self.loss(y_true, y_pred)
        reg_loss = tf.reduce_mean(self.losses)
        return (
            tf.reduce_mean(target_loss + self.penalty * count_loss),
            target_loss,
            count_loss,
            reg_loss,
        )

    def _compute_metric(self, y_true, outputs):
        _, _, y_pred, _ = outputs
        if self.num_classes is None:
            y, _ = y_true
            y = y * self.scale + self.shift
            y_pred = y_pred * self.scale + self.shift
            self.transfer_tracker(y, y_pred)
        else:
            y_true = tf.cast(y_true, dtype=tf.int32)
            y_true = tf.one_hot(y_true, depth=self.num_classes)
            self.transfer_tracker(y_true, y_pred)

    def build(self, input_shape=None):
        seq_input = tf.keras.Input([None, 150])
        count_input = tf.keras.Input([None])
        self.call([seq_input, count_input])
        super(TransferLearnNucleotideModel, self).build(
            [(None, None, 150), (None, None)]
        )

    def predict_step(self, data):
        inputs, y = data
        _, _, y_pred, _ = self(inputs, training=False)

        if self.num_classes is None:
            y_true, _ = y
            y_true = y_true * self.scale + self.shift
            y_pred = y_pred * self.scale + self.shift
            return y_pred, y_true
        return tf.argmax(y_pred, axis=-1), y

    def train_step(self, data):
        inputs, y = data
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, target_loss, count_mse, reg_loss = self._compute_loss(y, outputs)
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.target_tracker.update_state(target_loss)
        self.count_tracker.update_state(count_mse)
        self.reg_tracker.update_state(reg_loss)
        self._compute_metric(y, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "target_loss": self.target_tracker.result(),
            "count_mse": self.count_tracker.result(),
            self.transfer_string: self.transfer_tracker.result(),
        }

    def test_step(self, data):
        inputs, y = data

        outputs = self(inputs, training=False)
        loss, target_loss, count_mse, reg_loss = self._compute_loss(y, outputs)

        self.loss_tracker.update_state(loss)
        self.target_tracker.update_state(target_loss)
        self.count_tracker.update_state(count_mse)
        self.reg_tracker.update_state(reg_loss)
        self._compute_metric(y, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "target_loss": self.target_tracker.result(),
            "count_mse": self.count_tracker.result(),
            self.transfer_string: self.transfer_tracker.result(),
        }

    def call(self, inputs, training=False):
        tokens, counts = inputs

        # keras cast all input to float so we need to manually cast
        # to expected type
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=self.compute_dtype)
        extended_counts = tf.expand_dims(counts, axis=-1)

        count_mask = float_mask(extended_counts, dtype=self.compute_dtype)
        asv_embeddings = self.base_model(
            tokens,
            return_final_embeddings=True,
            randomly_mask_nucleotides=False,
            training=False,
        )
        asv_embeddings = tf.cast(asv_embeddings, dtype=self.compute_dtype)
        count_embeddings = self.count_encoder(
            extended_counts, count_mask=count_mask, training=training
        )
        count_embeddings = tf.cast(count_embeddings, dtype=self.compute_dtype)
        embeddings = count_embeddings

        # add <SAMPLE> token empbedding
        asv_shape = tf.shape(embeddings)
        batch_len = asv_shape[0]
        emb_len = asv_shape[-1]
        sample_emb_shape = [1 for _ in embeddings.get_shape().as_list()]
        sample_emb_shape[0] = batch_len
        sample_emb_shape[-1] = emb_len
        transfer_token = tf.broadcast_to(self.transfer_token, sample_emb_shape)
        transfer_token = tf.cast(transfer_token, dtype=self.compute_dtype)
        embeddings = tf.concat([embeddings, transfer_token], axis=1)

        # mask pad values out
        count_mask = tf.pad(count_mask, [[0, 0], [0, 1], [0, 0]], constant_values=1)

        # asv + count attention
        count_attention_mask = tf.matmul(count_mask, count_mask, transpose_b=True)
        cross_attention_mask = tf.linalg.band_part(count_attention_mask, -1, 0)

        embeddings = self.transfer_encoder(
            embeddings,
            asv_embeddings,
            self_attention_mask=cross_attention_mask,
            cross_attention_mask=count_attention_mask,
            training=training,
        )

        target_out = self.transfer_intermediate(embeddings[:, -1, :])
        target_out = self.transfer_ff(target_out)

        count_embeddings = embeddings[:, :-1, :]
        count_pred = self.count_intermediate(count_embeddings)
        count_pred = tf.squeeze(self.count_out(count_pred), axis=-1)
        return (
            count_embeddings,
            self.count_activation(count_pred),
            self.transfer_activation(target_out),
            counts,
        )

    def get_config(self):
        config = super(TransferLearnNucleotideModel, self).get_config()
        config.update(
            {
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "freeze_base_weights": self.freeze_base_weights,
                "mask_percent": self.mask_percent,
                "shift": self.shift,
                "scale": self.scale,
                "penalty": self.penalty,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["base_model"] = tf.keras.saving.deserialize_keras_object(
            config["base_model"]
        )
        model = cls(**config)
        return model
