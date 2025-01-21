from __future__ import annotations

import tensorflow as tf

from aam.losses import PairwiseLoss
from aam.models.base_sequence_encoder import BaseSequenceEncoder


@tf.keras.saving.register_keras_serializable(package="NucleotideEncoderV2")
class NucleotideEncoderV2(tf.keras.Model):
    def __init__(
        self,
        embedding_dim: int,
        max_bp: int,
        dropout_rate: float,
        intermediate_activation: str = "gelu",
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 256,
        normalize_outputs: bool = False,
        use_residual_connections: bool = False,
        regularize_embeddings=False,
        asv_encoder=None,
        use_linear_bias=False,
        **kwargs,
    ):
        super(NucleotideEncoderV2, self).__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.max_bp = max_bp
        self.dropout_rate = dropout_rate
        self.intermediate_activation = intermediate_activation
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.normalize_outputs = normalize_outputs
        self.use_residual_connections = use_residual_connections
        self.regularize_embeddings = regularize_embeddings

        self.loss_tracker = tf.keras.metrics.Mean()
        self.nuc_tracker = tf.keras.metrics.Mean()

        self.asv_loss = PairwiseLoss()
        self.asv_tracker = tf.keras.metrics.Mean()
        self.use_linear_bias = use_linear_bias
        self.asv_encoder = BaseSequenceEncoder(
            embedding_dim=self.embedding_dim,
            max_bp=self.max_bp,
            token_limit=2048,  # unused
            sample_attention_heads=self.attention_heads,
            sample_attention_layers=self.attention_layers,
            sample_intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            nuc_attention_heads=self.attention_heads,
            nuc_attention_layers=self.attention_layers,
            nuc_intermediate_size=self.intermediate_size,
            intermediate_activation=self.intermediate_activation,
            is_16S=True,
            nucleotide_encoder=None,
            normalize_outputs=self.normalize_outputs,
            use_residual_connections=self.use_residual_connections,
            asv_encoder=asv_encoder,
            use_linear_bias=self.use_linear_bias,
        )

    def build(self, input_shape):
        if self.built:
            return

        self.output_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        super(NucleotideEncoderV2, self).build(input_shape)

    def compile(self, include_bert_loss=True, **kwargs):
        super(NucleotideEncoderV2, self).compile(**kwargs)

        self.include_bert_loss = include_bert_loss

    def predict_step(self, data):
        inputs, asv_ids = data
        return self(inputs, training=False), asv_ids

    def _compute_loss(self, y_true, embeddings):
        num_pairs = tf.shape(y_true)[-1]
        embeddings = embeddings[:num_pairs]
        losses = self.asv_loss(y_true, embeddings)

        # # want "hard" examples
        # # pairwise loss seems to follow an expontial distribution
        # # the goal is to drive the mean towards 0
        # mean = tf.reduce_mean(losses)
        # mean_mask = losses >= mean
        # asv_loss = tf.reduce_mean(losses[mean_mask])
        # return tf.where(asv_loss > 0, asv_loss, 0.0)  # incase no loss are in the 75 quantile
        return tf.reduce_mean(losses)

    def train_step(self, data):
        inputs, y_true = data
        with tf.GradientTape() as tape:
            embeddings = self(inputs, training=True)
            asv_loss = self._compute_loss(y_true, embeddings)
            loss = asv_loss
            nuc_loss = tf.reduce_sum(self.losses)
            loss = nuc_loss + asv_loss

            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.nuc_tracker.update_state(nuc_loss)
        self.asv_tracker.update_state(asv_loss)
        return {
            "loss": self.loss_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "asv_loss": self.asv_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def test_step(self, data):
        inputs, y_true = data

        embeddings = self(inputs, training=False)
        asv_loss = self._compute_loss(y_true, embeddings)
        nuc_loss = tf.reduce_sum(self.losses)
        loss = asv_loss + nuc_loss
        self.loss_tracker.update_state(loss)
        self.nuc_tracker.update_state(nuc_loss)
        self.asv_tracker.update_state(asv_loss)
        return {
            "loss": self.loss_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "asv_loss": self.asv_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable
        tokens = inputs
        include_bert_loss = True
        if hasattr(self, "include_bert_loss"):
            include_bert_loss = self.include_bert_loss
        embeddings = self.asv_encoder(tokens, include_bert_random_mask=include_bert_loss, training=training)
        return self.output_activation(embeddings)

    def get_config(self):
        config = super(NucleotideEncoderV2, self).get_config()
        if hasattr(config, "include_bert_loss"):
            config.pop("include_bert_loss")
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "max_bp": self.max_bp,
                "dropout_rate": self.dropout_rate,
                "intermediate_activation": self.intermediate_activation,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "normalize_outputs": self.normalize_outputs,
                "use_residual_connections": self.use_residual_connections,
                "build_input_shape": self.get_build_config(),
                "regularize_embeddings": self.regularize_embeddings,
                "use_linear_bias": self.use_linear_bias,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)

        if input_shape is not None:
            model.build([None, 150])
        return model
