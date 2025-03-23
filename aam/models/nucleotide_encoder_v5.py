from __future__ import annotations

import tensorflow as tf

from aam.layers import ASVEncoder
from aam.losses import PairwiseLoss
from aam.models.base_sequence_encoder import BaseSequenceEncoder


@tf.keras.saving.register_keras_serializable(package="NucleotideEncoderV5")
class NucleotideEncoderV5(tf.keras.Model):
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
        use_linear_bias=False,
        **kwargs,
    ):
        super(NucleotideEncoderV5, self).__init__(**kwargs)
        print("Constructing V5 model")
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
        self.asv_loss = PairwiseLoss(use_mean_pairs=False)
        self.asv_tracker = tf.keras.metrics.Mean()

        self.use_linear_bias = use_linear_bias
        self.asv_encoder = ASVEncoder(
            self.max_bp,
            self.attention_heads,
            self.attention_layers,
            self.dropout_rate,
            self.intermediate_size,
            intermediate_activation=self.intermediate_activation,
            embedding_dim=self.embedding_dim,
            normalize_outputs=self.normalize_outputs,
            use_residual_connections=self.use_residual_connections,
            regularize_embeddings=self.regularize_embeddings,
            use_linear_bias=self.use_linear_bias,
            name="asv_encoder",
        )

        self.asv_ff_block = tf.keras.layers.Dense(
            self.embedding_dim,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.HeUniform(),
            dtype=tf.float32,
        )

    def build(self, input_shape):
        if self.built:
            return

        self.output_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        super(NucleotideEncoderV5, self).build(input_shape)

    def compile(self, include_bert_loss=True, **kwargs):
        super(NucleotideEncoderV5, self).compile(**kwargs)

        self.include_bert_loss = include_bert_loss

    def predict_step(self, data):
        inputs, asv_ids = data
        return self(inputs, training=False), asv_ids

    def _compute_loss(self, y_true, embeddings):
        loss = 0.0
        num_pairs = tf.shape(y_true)[-1]
        embeddings = embeddings[:num_pairs]
        asv_loss = tf.reduce_mean(self.asv_loss(y_true, embeddings))
        loss += asv_loss
        return loss, asv_loss

    def train_step(self, data):
        inputs, y_true = data
        with tf.GradientTape() as tape:
            embeddings = self(inputs, training=True)
            loss, asv_loss = self._compute_loss(y_true, embeddings)
            if self.include_bert_loss:
                nuc_loss = tf.reduce_sum(self.losses) * 0.1
            else:
                nuc_loss = 0.0
            loss += nuc_loss

            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.nuc_tracker.update_state(nuc_loss)
        self.asv_tracker.update_state(asv_loss)
        output_trackers = {
            "loss": self.loss_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "asv_loss": self.asv_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }
        return output_trackers

    def test_step(self, data):
        inputs, y_true = data

        embeddings = self(inputs, training=False)
        loss, asv_loss = self._compute_loss(y_true, embeddings)
        if self.include_bert_loss:
            nuc_loss = tf.reduce_sum(self.losses) * 0.1
        else:
            nuc_loss = 0.0
        loss += nuc_loss
        self.loss_tracker.update_state(loss)
        self.nuc_tracker.update_state(nuc_loss)
        self.asv_tracker.update_state(asv_loss)
        output_trackers = {
            "loss": self.loss_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "asv_loss": self.asv_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }
        return output_trackers

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable
        tokens = inputs
        include_bert_loss = False
        if hasattr(self, "include_bert_loss"):
            include_bert_loss = self.include_bert_loss
        embeddings = self.asv_encoder(
            tokens, include_bert_random_mask=include_bert_loss, training=training
        )

        asv_embeddings = tf.reduce_mean(embeddings, axis=1)
        embeddings = self.asv_ff_block(asv_embeddings, training=training)

        return self.output_activation(embeddings)

    def asv_embeddings(self, tokens):
        embeddings = self.asv_encoder(
            tokens, include_bert_random_mask=False, training=False
        )

        asv_embeddings = tf.reduce_mean(embeddings, axis=1)
        return self.asv_ff_block(asv_embeddings, training=False)

    def get_config(self):
        config = super(NucleotideEncoderV5, self).get_config()
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
