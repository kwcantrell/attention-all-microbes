from __future__ import annotations

from typing import Union

import tensorflow as tf
import tensorflow_models as tfm

from aam.losses import PairwiseLoss
from aam.models.base_sequence_encoder import BaseSequenceEncoder
from aam.models.multihead_attention_pooling import MultiHeadAttentionPooling
from aam.models.transformers import TransformerEncoder
from aam.models.utils import sort_using_counts, to_batch
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler
from aam.utils import float_mask


@tf.keras.saving.register_keras_serializable(package="UnifracEncoder")
class UnifracEncoder(tf.keras.layers.Layer):
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
        pairwise_loss_type="mse",
        normalize_outputs=False,
        use_residual_connections=True,
        use_residual_pool=None,
        **kwargs,
    ):
        super(UnifracEncoder, self).__init__(**kwargs)
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
        self.pairwise_loss_type = pairwise_loss_type
        self.normalize_outputs = normalize_outputs
        self.use_residual_connections = use_residual_connections

        if use_residual_pool is None:
            use_residual_pool = use_residual_connections
        self.use_residual_pool = use_residual_pool

    def build(self, input_shape):
        print(f"Input Shape in build: {input_shape}")
        if self.built:
            print("UnifracEncoder is already built")
            return

        self.attention_pooling = MultiHeadAttentionPooling(
            self.normalize_outputs, num_heads=self.attention_heads, use_residual_connections=self.use_residual_pool
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
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(self.token_limit, seq_axis=1)

        self.encoder_ff = tf.keras.layers.Dense(self.output_dim, dtype=tf.float32)
        super(UnifracEncoder, self).build(input_shape)

    @property
    def accumulation_steps(self):
        return self._accumulation_steps

    @accumulation_steps.setter
    def accumulation_steps(self, steps):
        self._accumulation_steps = steps
        self.gradient_accumulator = GradientAccumulator(self.accumulation_steps)

    def _unifrac_embeddings(self, tensor, mask=None, training=False):
        encoder_pred = self.attention_pooling(tensor, mask=mask, training=training)
        encoder_pred = self.encoder_ff(encoder_pred)
        return encoder_pred

    def call(
        self,
        inputs,
        attention_mask=True,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable

        asv_embeddings = inputs + tf.cast(self._rezero, dtype=self.compute_dtype) * self.pos_emb(inputs)
        asv_embeddings = self.encoder(asv_embeddings, mask=attention_mask, training=training)
        unifrac_embeddings = self._unifrac_embeddings(asv_embeddings, attention_mask, training=training)

        print("UnifracEncoder exit...", self.trainable)
        return asv_embeddings, unifrac_embeddings

    def base_embeddings(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        sample_embeddings = self.base_encoder.base_embeddings(tokens)

        # account for <SAMPLE> token
        count_mask = float_mask(counts, dtype=tf.int32)
        count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        count_attention_mask = count_mask

        unifrac_gated_embeddings = self.unifrac_encoder(sample_embeddings, mask=count_attention_mask)
        unifrac_pred = unifrac_gated_embeddings[:, 0, :]
        unifrac_pred = self.unifrac_ff(unifrac_pred)

        unifrac_embeddings = sample_embeddings + unifrac_gated_embeddings * self._unifrac_alpha

        return unifrac_embeddings

    def asv_embeddings(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens = inputs
        sample_embeddings = self.base_encoder(tokens, training=False)
        return sample_embeddings

    def asv_gradient(self, inputs: tuple[tf.Tensor, tf.Tensor], asv_embeddings) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        sample_embeddings = self.base_encoder.asv_gradient(tokens, asv_embeddings)

        # account for <SAMPLE> token
        count_mask = float_mask(counts, dtype=tf.int32)
        count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        count_attention_mask = count_mask

        unifrac_gated_embeddings = self.unifrac_encoder(sample_embeddings, mask=count_attention_mask)
        unifrac_pred = unifrac_gated_embeddings[:, 0, :]
        unifrac_pred = self.unifrac_ff(unifrac_pred)

        unifrac_embeddings = sample_embeddings + unifrac_gated_embeddings * self._unifrac_alpha

        return unifrac_embeddings

    @property
    def train_nuc_encoder(self):
        return self.base_encoder.trainable

    @train_nuc_encoder.setter
    def train_nuc_encoder(self, flag: bool):
        print("train nuc encoder:", flag)
        self.base_encoder.trainable = flag

    def get_config(self):
        config = super(UnifracEncoder, self).get_config()
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
                "normalize_outputs": self.normalize_outputs,
                "use_residual_connections": self.use_residual_connections,
                "use_residual_pool": self.use_residual_pool,
            }
        )
        return config
