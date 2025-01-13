from __future__ import annotations

from typing import Union

import tensorflow as tf
import tensorflow_models as tfm

from aam.losses import PairwiseLoss
from aam.models.transformers import TransformerEncoder
from aam.models.utils import sort_using_counts, to_batch
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler
from aam.utils import float_mask


@tf.keras.saving.register_keras_serializable(package="CountEncoder")
class CountEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        output_dim: int,
        token_limit: int,
        encoder_type: str,
        dropout_rate: float = 0.0,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 1024,
        intermediate_activation: str = "gelu",
        max_bp: int = 150,
        is_16S: bool = True,
        vocab_size: int = 6,
        add_token: bool = True,
        asv_dropout_rate: float = 0.0,
        accumulation_steps: int = 1,
        nucleotide_encoder=None,
        pairwise_loss_type="mse",
        normalize_outputs=False,
        use_residual_connections=True,
        **kwargs,
    ):
        super(CountEncoder, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.token_limit = token_limit
        self.encoder_type = encoder_type
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.max_bp = max_bp
        self.is_16S = is_16S
        self.vocab_size = vocab_size
        self.add_token = add_token
        self.asv_dropout_rate = asv_dropout_rate
        self.accumulation_steps = accumulation_steps
        self.nucleotide_encoder = nucleotide_encoder
        self.pairwise_loss_type = pairwise_loss_type
        self.normalize_outputs = normalize_outputs
        self.use_residual_connections = use_residual_connections

    def build(self, input_shape):
        print(f"Input Shape in build: {input_shape}")
        if self.built:
            print("already built")
            return
        # layers used in model
        hidden_dim = self.embedding_dim
        key_dim = int(hidden_dim // self.attention_heads)
        self.attention = tf.keras.layers.MultiHeadAttention(
            self.attention_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
        )
        self.encoder = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
            normalize_outputs=self.normalize_outputs,
            use_residual_connections=self.use_residual_connections,
            name="encoder",
        )

        self._rezero = self.add_weight(
            name="rezero_alpha", initializer=tf.keras.initializers.Zeros(), trainable=True, dtype=tf.float32
        )
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(
            self.token_limit, seq_axis=1, initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
        )

        self.encoder_ff = tf.keras.layers.Dense(1)
        self._softmax = tf.keras.layers.Activation("softmax", dtype=tf.float32)
        self._softmax.build(input_shape)
        super(CountEncoder, self).build(input_shape)

    def _compute_loss(
        self,
        counts: tuple[tf.Tensor, tf.Tensor],
        count_embeddings: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        count_pred = self.encoder_ff(count_embeddings)
        count_pred = self._softmax(count_pred)
        count_mask = tf.cast(counts > 0, dtype=tf.float32)
        count_loss = tf.square(tf.math.log1p(counts) - tf.math.log1p(count_pred)) * count_mask
        self.add_loss(tf.reduce_mean(count_loss))

    def _relative_abundance(self, counts: tf.Tensor) -> tf.Tensor:
        count_sums = tf.reduce_sum(counts, axis=1, keepdims=True)
        rel_abundance = counts / count_sums
        return rel_abundance

    def call(
        self,
        inputs,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        embeddings, counts = inputs
        counts = tf.cast(counts, dtype=tf.float32)
        count_mask = tf.cast(counts > 0, dtype=self.compute_dtype)
        rel_abundance = self._relative_abundance(counts)

        pos_embeddings = self.pos_emb(embeddings) * tf.cast((rel_abundance), dtype=self.compute_dtype)
        attention_mask = tf.matmul(count_mask, count_mask, transpose_b=True)
        pos_embeddings = self.attention(pos_embeddings, pos_embeddings, attention_mask=attention_mask, training=training)
        self._compute_loss(rel_abundance, pos_embeddings, training=training)

        count_embeddings = embeddings + tf.cast(self._rezero, dtype=self.compute_dtype) * pos_embeddings
        count_embeddings = self.encoder(count_embeddings, count_mask, training=training)
        print("CountEncoder exit...")
        return count_embeddings

    def get_config(self):
        config = super(CountEncoder, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "token_limit": self.token_limit,
                "encoder_type": self.encoder_type,
                "dropout_rate": self.dropout_rate,
                "embedding_dim": self.embedding_dim,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "intermediate_activation": self.intermediate_activation,
                "max_bp": self.max_bp,
                "is_16S": self.is_16S,
                "vocab_size": self.vocab_size,
                "add_token": self.add_token,
                "asv_dropout_rate": self.asv_dropout_rate,
                "accumulation_steps": self.accumulation_steps,
                "nucleotide_encoder": self.nucleotide_encoder,
                "normalize_outputs": self.normalize_outputs,
                "use_residual_connections": self.use_residual_connections,
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)

        if input_shape is not None:
            model.build(input_shape)
        return model
