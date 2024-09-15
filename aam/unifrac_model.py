import tensorflow as tf
from aam.utils import float_mask

from aam.layers import (
    NucleotideEmbedding,
    ReadHead,
)
from aam.losses import PairwiseLoss


@tf.keras.saving.register_keras_serializable(package="UnifracModel")
class UnifracModel(tf.keras.Model):
    def __init__(
        self,
        token_dim,
        max_bp,
        pca_hidden_dim,
        pca_heads,
        pca_layers,
        attention_heads,
        attention_layers,
        attention_ff,
        dropout_rate,
        **kwargs,
    ):
        super(UnifracModel, self).__init__(**kwargs)
        self.token_dim = token_dim
        self.max_bp = max_bp
        self.pca_hidden_dim = pca_hidden_dim
        self.pca_heads = pca_heads
        self.pca_layers = pca_layers
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.attention_ff = attention_ff
        self.dropout_rate = dropout_rate
        self.regresssion_loss = PairwiseLoss()
        self.attention_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction="none"
        )
        self.loss_tracker = tf.keras.metrics.Mean()
        self.metric_traker = tf.keras.metrics.Mean()
        self.confidence_tracker = tf.keras.metrics.CategoricalAccuracy()

        # layers used in model
        self.feature_emb = NucleotideEmbedding(
            token_dim,
            max_bp,
            pca_hidden_dim,
            pca_heads,
            pca_layers,
            attention_heads,
            attention_layers,
            attention_ff,
            dropout_rate,
        )
        self.readhead = ReadHead(
            max_bp=max_bp,
            hidden_dim=pca_hidden_dim,
            num_heads=pca_heads,
            num_layers=pca_layers,
            output_dim=32,
            output_nuc=True,
            name="unifrac_output",
        )
        self.linear_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        self.nucleotide_position = tf.range(0, 4 * 150, 4, dtype=tf.int32)

        def _compute_loss(target, outputs):
            reg_out, logits, tokens = outputs

            # Compute regression loss
            reg_loss = tf.reduce_mean(self.regresssion_loss(target, reg_out))

            loss = reg_loss
            token_cat = tf.one_hot(tokens, depth=6)  # san -> san6

            asv_mask = float_mask(
                tf.reduce_sum(tokens, axis=-1, keepdims=True)
            )  # san -> sa1

            asv_counts = float_mask(tf.reduce_sum(tokens, axis=-1))  # san -> sa
            asv_counts = asv_counts = tf.reduce_sum(
                asv_counts, axis=-1, keepdims=True
            )  # sa -> s1)

            # nucleotide level
            asv_loss = self.attention_loss(token_cat, logits)  # san6 -> san
            asv_loss = asv_loss * asv_mask
            asv_loss = tf.reduce_sum(asv_loss, axis=-1)  # san -> sa

            # asv_level
            asv_loss = tf.reduce_sum(asv_loss, axis=-1, keepdims=True)  # sa -> s1
            asv_loss = asv_loss / asv_counts

            # total
            loss = loss + tf.reduce_sum(asv_loss)
            return [loss, reg_loss, asv_loss]

        # def compute_metrics(reg_out, att_out, tokens)
        self._compute_loss = _compute_loss

    def call(self, inputs, training=False):
        tokens, counts = inputs
        token_mask = float_mask(tokens, dtype=tf.int32)
        features = tokens + tf.reshape(self.nucleotide_position, shape=[1, 1, -1])
        features = tf.multiply(features, token_mask)

        inputs = (features, counts)
        feature_emb_outputs = self.feature_emb(
            inputs,
            training=training,
        )
        (reg_out, att_out) = self.readhead(feature_emb_outputs, training=training)
        reg_out = self.linear_activation(reg_out)
        return reg_out, att_out, tokens

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
        super(UnifracModel, self).build(
            (tf.TensorShape([None, None, 150]), tf.TensorShape([None, None]))
        )

    def train_step(self, data):
        inputs, y = data
        with tf.GradientTape() as tape:
            reg_out, logits, tokens = self(inputs, training=True)
            loss, reg_loss, asv_loss = self._compute_loss(y, [reg_out, logits, tokens])
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.metric_traker.update_state(reg_loss)

        tokens = tf.one_hot(tokens, depth=6)
        self.confidence_tracker.update_state(tokens, logits)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.metric_traker.result(),
            "confidence": self.confidence_tracker.result(),
        }

    def test_step(self, data):
        inputs, y = data
        reg_out, logits, tokens = self(inputs, training=True)
        loss, reg_loss, asv_loss = self._compute_loss(y, [reg_out, logits, tokens])

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.metric_traker.update_state(reg_loss)

        tokens = tf.one_hot(tokens, depth=6)
        self.confidence_tracker.update_state(tokens, logits)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.metric_traker.result(),
            "confidence": self.confidence_tracker.result(),
        }

    def predict_step(self, data):
        inputs, y = data
        # Forward pass
        reg_out, _, _ = self(inputs, training=False)
        return tf.squeeze(reg_out), tf.squeeze(y)

    @tf.function
    def sequence_embedding(self, seq, squeeze=True):
        """returns sequenuce embeddings

        Args:
            seq (StringTensor): ASV sequences.

        Returns:
            _type_: _description_
        """
        seq = tf.expand_dims(seq, axis=0)
        token_mask = float_mask(seq, dtype=tf.int32)
        features = seq + tf.reshape(self.nucleotide_position, shape=[1, 1, -1])
        features = tf.multiply(features, token_mask)
        sequence_embedding = self.feature_emb.sequence_embedding(seq)
        if squeeze:
            return tf.squeeze(sequence_embedding, axis=0)
        else:
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
        sequence_embedding = self.feature_emb.sequence_embedding(seq)

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
                "pca_hidden_dim": self.pca_hidden_dim,
                "pca_heads": self.pca_heads,
                "pca_layers": self.pca_layers,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "attention_ff": self.attention_ff,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
