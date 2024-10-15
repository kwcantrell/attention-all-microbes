from __future__ import annotations

import tensorflow as tf

from aam.layers import (
    ASVEncoder,
    SampleEncoder,
)
from aam.losses import PairwiseLoss
from aam.utils import apply_random_mask, float_mask, masked_loss


@tf.keras.saving.register_keras_serializable(package="UnifracModel")
class UnifracModel(tf.keras.Model):
    def __init__(
        self,
        token_dim,
        max_bp,
        attention_heads,
        attention_layers,
        attention_ff,
        dropout_rate,
        penalty=0.01,
        nuc_attention_heads=2,
        nuc_attention_layers=4,
        intermediate_ff=1024,
        **kwargs,
    ):
        super(UnifracModel, self).__init__(**kwargs)
        self.token_dim = token_dim
        self.max_bp = max_bp
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.attention_ff = attention_ff
        self.dropout_rate = dropout_rate
        self.penalty = penalty
        self.nuc_attention_heads = nuc_attention_heads
        self.nuc_attention_layers = nuc_attention_layers
        self.intermediate_ff = intermediate_ff
        self.regresssion_loss = PairwiseLoss()
        self.attention_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            ignore_class=0, from_logits=False, reduction="none"
        )
        self.loss_tracker = tf.keras.metrics.Mean()
        self.metric_traker = tf.keras.metrics.Mean()
        self.entropy = tf.keras.metrics.Mean()
        self.accuracy = tf.keras.metrics.Mean()

        # layers used in model
        self.asv_encoder = ASVEncoder(
            token_dim,
            max_bp,
            nuc_attention_heads,
            nuc_attention_layers,
            dropout_rate,
            intermediate_ff,
            name="asv_encoder",
        )
        self.sample_encoder = SampleEncoder(
            token_dim,
            max_bp,
            attention_heads,
            attention_layers,
            attention_ff,
            dropout_rate,
            name="sample_encoder",
        )
        self.nuc_logits = tf.keras.layers.Dense(
            6, use_bias=False, name="nuc_logits", dtype=tf.float32, activation="softmax"
        )

        self.linear_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)

    @masked_loss(sparse_cat=True)
    def _compute_nuc_loss(self, tokens, pred_tokens):
        return self.attention_loss(tokens, pred_tokens)

    def _compute_loss(self, target, outputs):
        sample_embeddings, logits, tokens = outputs

        # Compute regression loss
        reg_loss = self.regresssion_loss(target, sample_embeddings)
        num_samples = tf.reduce_sum(float_mask(reg_loss))
        reg_loss = tf.math.divide_no_nan(tf.reduce_sum(reg_loss), num_samples)

        asv_loss = self._compute_nuc_loss(tokens, logits)

        # total
        loss = reg_loss + asv_loss
        return [loss, reg_loss, asv_loss]

    def _compute_accuracy(self, y_true, y_pred):
        tokens = tf.cast(y_true, dtype=tf.float32)

        pred_classes = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.float32)
        accuracy = tf.cast(tf.equal(tokens, pred_classes), dtype=tf.float32)

        mask = float_mask(tokens)
        accuracy = accuracy * mask

        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def call(
        self,
        inputs,
        return_nuc_embeddings=False,
        randomly_mask_nucleotides=True,
        training=False,
    ):
        # need to cast inputs to int32 to avoid error
        # because keras converts all inputs
        # to float when calling build()
        inputs = tf.cast(inputs, dtype=tf.int32)

        if training:
            inputs = apply_random_mask(inputs, 0.1)

        embeddings = self.asv_encoder(
            inputs,
            training=training,
        )
        asv_embeddings = embeddings[:, :, -1, :]

        if randomly_mask_nucleotides and training:
            asv_embeddings = apply_random_mask(asv_embeddings, 0.1)

        asv_mask = float_mask(tf.reduce_sum(inputs, axis=-1, keepdims=True))
        sample_embeddings = self.sample_encoder(
            asv_embeddings, attention_mask=asv_mask, training=training
        )

        if return_nuc_embeddings:
            nuc_embeddings = embeddings[:, :, :-1, :]
            nucleotides = self.nuc_logits(nuc_embeddings)
            return sample_embeddings, nucleotides

        return sample_embeddings

    def build(self, input_shape=None):
        super(UnifracModel, self).build(tf.TensorShape([None, None, self.max_bp]))

    def train_step(self, data):
        inputs, y = data
        with tf.GradientTape() as tape:
            _, sample_embeddings, logits, tokens = self(inputs, training=True)
            loss, reg_loss, asv_loss = self._compute_loss(
                y, [sample_embeddings, logits, tokens]
            )
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.metric_traker.update_state(reg_loss)
        self.entropy.update_state(asv_loss)

        self.accuracy.update_state(self._compute_accuracy(tokens, logits))
        return {
            "loss": self.loss_tracker.result(),
            "mse": self.metric_traker.result(),
            "entropy": self.entropy.result(),
            "accuracy": self.accuracy.result(),
        }

    def test_step(self, data):
        inputs, y = data
        _, sample_embeddings, logits, tokens = self(inputs, training=True)
        loss, reg_loss, asv_loss = self._compute_loss(
            y, [sample_embeddings, logits, tokens]
        )

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.metric_traker.update_state(reg_loss)
        self.entropy.update_state(asv_loss)

        self.accuracy.update_state(self._compute_accuracy(tokens, logits))
        return {
            "loss": self.loss_tracker.result(),
            "mse": self.metric_traker.result(),
            "entropy": self.entropy.result(),
            "accuracy": self.accuracy.result(),
        }

    def predict_step(self, data):
        inputs, y = data
        # Forward pass
        _, sample_embeddings, _, _ = self(inputs, training=False)
        return tf.squeeze(sample_embeddings), tf.squeeze(y)

    def sequence_embedding(self, seq):
        """returns sequenuce embeddings

        Args:
            seq (Tensor): ASV sequences. [batch, 150]
        Returns:
            _type_: _description_
        """
        seq = tf.expand_dims(seq, axis=1)
        sequence_embedding = self.asv_encoder(seq)
        sequence_embedding = sequence_embedding[:, :, -1, :]
        sequence_embedding = tf.squeeze(sequence_embedding)
        return sequence_embedding

    def edit_distance(self, seq):
        """computes the edit distance between true seq and AM^-1

        Args:
            seq (StringTensor): ASV sequences.


        Returns:
            _type_: _description_
        """
        seq = tf.cast(self.sequence_tokenizer(seq), tf.int32)
        seq = tf.expand_dims(seq, axis=0)
        sequence_embedding = self.asv_encoder.sequence_embedding(seq)

        sequence_logits = self.readhead.sequence_logits(sequence_embedding)
        pred_seq = tf.argmax(
            tf.nn.softmax(sequence_logits, axis=-1), axis=-1, output_type=tf.int32
        )
        sequence_mismatch_tokens = tf.not_equal(pred_seq, seq)
        sequence_mismatch_counts = tf.reduce_sum(
            tf.cast(sequence_mismatch_tokens, dtype=tf.int32), axis=-1
        )
        mask = tf.reshape(tf.not_equal(seq[:, :, 0], 0), shape=(-1,))
        return tf.reshape(sequence_mismatch_counts, shape=(-1,))[mask]

    def get_config(self):
        config = super(UnifracModel, self).get_config()
        config.update(
            {
                "token_dim": self.token_dim,
                "max_bp": self.max_bp,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "attention_ff": self.attention_ff,
                "dropout_rate": self.dropout_rate,
                "penalty": self.penalty,
                "nuc_attention_heads": self.nuc_attention_heads,
                "nuc_attention_layers": self.nuc_attention_layers,
                "intermediate_ff": self.intermediate_ff,
            }
        )
        return config
