import tensorflow as tf
from tensorflow._api.v2.math import reduce_sum
import tensorflow_models as tfm
from aam.utils import float_mask
from aam.losses import PairwiseLoss


@tf.keras.saving.register_keras_serializable(package="UnifracModel")
class GOTURegressoin(tf.keras.Model):
    def __init__(
        self,
        total_tokens,
        max_token_per_sample,
        emb_dim,
        batch_size=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.total_tokens = total_tokens
        self.max_token_per_sample = max_token_per_sample
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        # self.regression_loss = tf.keras.losses.MeanSquaredError()
        self.regression_loss = PairwiseLoss()
        self.attention_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        self.loss_tracker = tf.keras.metrics.Mean()
        self.reg_tracker = tf.keras.metrics.Mean()
        self.attention_tracker = tf.keras.metrics.SparseCategoricalAccuracy()

        # layers used in model
        self.feature_emb = tf.keras.layers.Embedding(total_tokens, emb_dim)
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(max_token_per_sample + 1)
        self.encoder = tfm.nlp.models.TransformerEncoder(dropout_rate=0.1)
        self.att_out = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(emb_dim, activation="relu"),
                tf.keras.layers.Dense(emb_dim, activation="relu"),
                tf.keras.layers.Dense(total_tokens),
            ]
        )

    def call(self, inputs, return_transfer_info=False, training=False):
        """inputs: [(B, N), (B, N)]"""
        features_pre_pad, counts = inputs
        counts = tf.cast(counts, dtype=self.compute_dtype)
        features = tf.pad(
            features_pre_pad, [[0, 0], [0, 1]], constant_values=self.total_tokens - 1
        )
        counts = tf.pad(counts, [[0, 0], [0, 1]], constant_values=0)

        attention_mask = float_mask(tf.expand_dims(features, axis=-1))
        attention_mask = tf.matmul(attention_mask, attention_mask, transpose_b=True)

        if training:
            seq_mask = tf.random.uniform(
                tf.shape(counts), minval=0, maxval=1, dtype=self.compute_dtype
            )
            seq_mask = tf.less_equal(seq_mask, 0.9)
            seq_mask = tf.cast(seq_mask, dtype=self.compute_dtype)
            features = tf.multiply(features, tf.cast(seq_mask, dtype=tf.int64))

        embeddings = self.feature_emb(features)
        counts = tf.expand_dims(counts, axis=-1)
        embeddings = embeddings + self.pos_emb(embeddings)

        output = self.encoder(
            embeddings, attention_mask=attention_mask, training=training
        )
        if return_transfer_info:
            return embeddings, features_pre_pad, counts, attention_mask
        else:
            reg_out = output[:, -1, :]
            att_out = self.att_out(output[:, :-1, :], training=training)
            return (reg_out, att_out, features_pre_pad)

    def predict_step(self, data):
        inputs, y = data
        reg_out, logits, features = self(inputs, training=False)
        return reg_out

    def train_step(self, data):
        """
        Need to add checks to verify input/output tuples are formatted correctly
        """
        inputs, y = data
        with tf.GradientTape() as tape:
            # Forward pass
            reg_out, logits, features = self(inputs, training=True)
            # Compute regression loss
            reg_loss = tf.reduce_sum(self.regression_loss(y, reg_out), axis=-1)
            loss = tf.reduce_mean(reg_loss)

            attention_loss = tf.reduce_sum(
                self.attention_loss(features, logits), axis=-1
            )
            loss += tf.reduce_mean(attention_loss)

        # # Update weights
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.reg_tracker.update_state(reg_loss)
        self.attention_tracker(tf.expand_dims(features, axis=-1), logits)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.reg_tracker.result(),
            "attention_accuracy": self.attention_tracker.result(),
        }

    def test_step(self, data):
        """
        Need to add checks to verify input/output tuples are formatted correctly
        """
        inputs, y = data

        # Forward pass
        reg_out, logits, features = self(inputs, training=False)
        # Forward pass
        reg_loss = tf.reduce_sum(self.regression_loss(y, reg_out), axis=-1)
        loss = tf.reduce_mean(reg_loss)

        attention_loss = tf.reduce_sum(self.attention_loss(features, logits), axis=-1)
        loss += tf.reduce_mean(attention_loss)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.reg_tracker.update_state(reg_loss)
        self.attention_tracker(tf.expand_dims(features, axis=-1), logits)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.reg_tracker.result(),
            "attention_accuracy": self.attention_tracker.result(),
        }

    def get_config(self):
        base_config = super().get_config()
        config = {
            "total_tokens": self.total_tokens,
            "max_token_per_sample": self.max_token_per_sample,
            "emb_dim": self.emb_dim,
            "batch_size": self.batch_size,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(package="GOTUTransfer")
class GOTUTransfer(tf.keras.Model):
    def __init__(
        self,
        base_model,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.regression_loss = tf.keras.losses.MeanSquaredError(reduction="none")
        self.attention_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        self.loss_tracker = tf.keras.metrics.Mean()
        self.reg_tracker = tf.keras.metrics.MeanAbsoluteError()
        self.attention_tracker = tf.keras.metrics.SparseCategoricalAccuracy()

        # layers used in model
        self.encoder = tfm.nlp.models.TransformerEncoder()
        # self.count_ff = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Dense(self.base_model.emb_dim, activation="relu"),
        #         tf.keras.layers.Dense(self.base_model.emb_dim, activation="relu"),
        #         tf.keras.layers.Dense(self.base_model.emb_dim),
        #     ]
        # )

        self.att_out = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.base_model.emb_dim, activation="relu"),
                tf.keras.layers.Dense(self.base_model.emb_dim, activation="relu"),
                tf.keras.layers.Dense(self.base_model.total_tokens),
            ]
        )
        self.reg_out = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        """inputs: [(B, N), (B, N)]"""
        embeddings, features_pre_pad, counts, attention_mask = self.base_model(
            inputs, return_transfer_info=True, training=False
        )
        # embeddings = embeddings + self.count_ff(counts, training=training)
        embeddings = embeddings + counts
        embeddings = self.encoder(
            embeddings, attention_mask=attention_mask, training=False
        )
        reg_out = self.reg_out(embeddings[:, -1, :])
        att_out = self.att_out(embeddings[:, :-1, :])
        return reg_out, att_out, features_pre_pad

    def predict_step(self, data):
        inputs, y = data
        reg_out, logits, features = self(inputs, training=False)
        return reg_out

    def train_step(self, data):
        """
        Need to add checks to verify input/output tuples are formatted correctly
        """
        inputs, y = data
        with tf.GradientTape() as tape:
            # Forward pass
            reg_out, logits, features = self(inputs, training=True)
            # Compute regression loss
            reg_loss = tf.reduce_sum(self.regression_loss(y, reg_out), axis=-1)
            loss = tf.reduce_mean(reg_loss)

            attention_loss = tf.reduce_sum(
                self.attention_loss(features, logits), axis=-1
            )
            loss += tf.reduce_mean(attention_loss)

        # # Update weights
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.reg_tracker.update_state(y, reg_out)
        self.attention_tracker(tf.expand_dims(features, axis=-1), logits)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.reg_tracker.result(),
            "attention_accuracy": self.attention_tracker.result(),
        }

    def test_step(self, data):
        """
        Need to add checks to verify input/output tuples are formatted correctly
        """
        inputs, y = data

        # Forward pass
        reg_out, logits, features = self(inputs, training=False)
        # Forward pass
        reg_loss = tf.reduce_sum(self.regression_loss(y, reg_out), axis=-1)
        loss = tf.reduce_mean(reg_loss)

        attention_loss = tf.reduce_sum(self.attention_loss(features, logits), axis=-1)
        loss += tf.reduce_mean(attention_loss)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.reg_tracker.update_state(y, reg_out)
        self.attention_tracker(tf.expand_dims(features, axis=-1), logits)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.reg_tracker.result(),
            "attention_accuracy": self.attention_tracker.result(),
        }

    def get_config(self):
        base_config = super().get_config()
        config = {
            "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
        }
        return {**base_config, **config}
