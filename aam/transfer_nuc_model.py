import tensorflow as tf
import tensorflow_models as tfm

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
        self.transfer_token = self.add_weight(
            "transfer_token",
            [1, 1, 128],
            dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )

        self.count_ff_inner = tf.keras.layers.Dense(
            128, activation="relu", dtype=tf.float32
        )
        self.count_ff_outer = tf.keras.layers.Dense(128, dtype=tf.float32)
        self.count_out = tf.keras.layers.Dense(1, use_bias=True, dtype=tf.float32)

        self.pos_embeddings = tfm.nlp.layers.RelativePositionEmbedding(
            128, dtype=tf.float32
        )
        self.count_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        self.count_tracker = tf.keras.metrics.Mean()

        self.transfer_encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            dropout_rate=self.dropout,
            activation="relu",
        )

        self.transfer_inner = tf.keras.layers.Dense(
            128, activation="relu", dtype=tf.float32
        )
        self.transfer_outer = tf.keras.layers.Dense(128, dtype=tf.float32)
        if self.num_classes is None:
            self.transfer_ff = tf.keras.layers.Dense(1, use_bias=True, dtype=tf.float32)
            self.transfer_tracker = tf.keras.metrics.MeanAbsoluteError()
            self.transfer2_tracker = tf.keras.metrics.MeanSquaredError()
            self.transfer_string = "mae"
            self.transfer_activation = tf.keras.layers.Activation(
                "linear", dtype=tf.float32
            )
        else:
            self.transfer_ff = tf.keras.layers.Dense(
                num_classes, use_bias=False, dtype=tf.float32
            )
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
        y, _ = y_true
        _, count_pred, y_pred, counts = outputs

        # count mask
        count_mask = float_mask(counts)
        count_mask = tf.ensure_shape(count_mask, [None, None])
        count_pred = tf.ensure_shape(count_pred, [None, None])

        # count mse
        relative_counts = self._relative_abundance(counts)
        count_loss = tf.math.square(relative_counts - count_pred) * count_mask
        count_loss = tf.reduce_sum(count_loss) / tf.reduce_sum(count_mask)

        # target loss
        y_pred = tf.ensure_shape(y_pred, [None, 1])
        target_loss = tf.reduce_mean(self.loss(y, y_pred))

        # regularization loss
        reg_loss = tf.reduce_mean(self.losses)
        return (
            target_loss + count_loss,
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
            self.transfer2_tracker(y, y_pred)
        else:
            y_true = tf.cast(y_true, dtype=tf.int32)
            y_true = tf.one_hot(y_true, depth=self.num_classes)
            self.transfer_tracker(y_true, y_pred)

    def build(self, input_shape=None):
        seq_input = tf.keras.Input([None, 150], dtype=tf.int32)
        count_input = tf.keras.Input([None], dtype=tf.int32)
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
            "mse": self.transfer2_tracker.result(),
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
            "mse": self.transfer2_tracker.result(),
        }

    def _pos_embeddings(self, asv_embeddings):
        num_asvs = tf.shape(asv_embeddings)[1] - 1
        pos_embeddings = tf.expand_dims(
            self.pos_embeddings(None, length=num_asvs), axis=0
        )
        pos_embeddings = tf.pad(
            pos_embeddings, [[0, 0], [0, 1], [0, 0]], constant_values=0
        )
        return pos_embeddings

    def _relative_abundance(self, counts, add_pad=False):
        counts = tf.cast(counts, dtype=tf.float32)
        count_sums = tf.reduce_sum(counts, axis=1, keepdims=True)
        rel_abundance = counts / count_sums
        if add_pad:
            rel_abundance = tf.pad(
                rel_abundance, [[0, 0], [0, 1], [0, 0]], constant_values=1
            )
        return rel_abundance

    def call(self, inputs, training=False):
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        extended_counts = tf.expand_dims(counts, axis=-1)
        count_mask = float_mask(extended_counts, dtype=tf.int32)
        asv_embeddings = self.base_model(
            tokens,
            return_final_embeddings=True,
            randomly_mask_nucleotides=False,
            training=False,
        )

        pos_embeddings = self._pos_embeddings(asv_embeddings)
        embeddings = asv_embeddings + pos_embeddings

        rel_abundance = self._relative_abundance(extended_counts, add_pad=True)

        # randomly mask 10%
        rel_mask = tf.ones_like(rel_abundance, dtype=tf.float32)
        if training:
            random_mask = tf.random.uniform(
                tf.shape(rel_abundance), minval=0, maxval=1, dtype=self.compute_dtype
            )
            random_mask = tf.greater_equal(random_mask, 0.1)
            rel_mask = rel_mask * tf.cast(random_mask, dtype=tf.float32)

        # need to set all zeros to 1 to avoid setting asv_embeddings to zero
        rel_abundance = rel_abundance * rel_mask
        rel_abundance = rel_abundance + (1 - float_mask(rel_abundance))
        count_embeddings = asv_embeddings * rel_abundance

        # add <SAMPLE> token empbedding
        asv_shape = tf.shape(count_embeddings)
        batch_len = asv_shape[0]
        emb_len = asv_shape[-1]
        sample_emb_shape = [1 for _ in count_embeddings.get_shape().as_list()]
        sample_emb_shape[0] = batch_len
        sample_emb_shape[-1] = emb_len
        transfer_token = tf.broadcast_to(self.transfer_token, sample_emb_shape)
        transfer_token = tf.cast(transfer_token, dtype=tf.float32)
        embeddings = tf.concat([count_embeddings, transfer_token], axis=1)

        # mask pad values out
        count_mask = tf.pad(count_mask, [[0, 0], [0, 2], [0, 0]], constant_values=1)

        # asv + count attention
        count_attention_mask = tf.matmul(count_mask, count_mask, transpose_b=True)

        embeddings = self.transfer_encoder(
            embeddings,
            attention_mask=count_attention_mask > 0,
            training=training,
        )

        target_out = self.transfer_inner(embeddings[:, -1, :])
        target_out = self.transfer_outer(target_out)
        target_out = self.transfer_ff(target_out)

        count_embeddings = self.count_ff_inner(embeddings[:, :-2, :])
        count_pred = self.count_ff_outer(count_embeddings)
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
        config["penalty"] = tf.keras.saving.deserialize_keras_object(config["penalty"])
        model = cls(**config)
        return model
