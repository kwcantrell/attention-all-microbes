from __future__ import annotations

import tensorflow as tf
import tensorflow_models as tfm

from aam.layers import (
    ASVEncoder,
)
from aam.models.transformers import TransformerEncoder
from aam.utils import float_mask, masked_loss


@tf.keras.saving.register_keras_serializable(package="BaseSequenceEncoder")
class BaseSequenceEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim: int,
        max_bp: int,
        token_limit: int,
        sample_attention_heads: int,
        sample_attention_layers: int,
        sample_intermediate_size: int,
        dropout_rate: float,
        nuc_attention_heads: int = 2,
        nuc_attention_layers: int = 4,
        nuc_intermediate_size: int = 1024,
        intermediate_activation: str = "gelu",
        **kwargs,
    ):
        super(BaseSequenceEncoder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.max_bp = max_bp
        self.token_limit = token_limit
        self.sample_attention_heads = sample_attention_heads
        self.sample_attention_layers = sample_attention_layers
        self.sample_intermediate_size = sample_intermediate_size
        self.dropout_rate = dropout_rate
        self.nuc_attention_heads = nuc_attention_heads
        self.nuc_attention_layers = nuc_attention_layers
        self.nuc_intermediate_size = nuc_intermediate_size
        self.intermediate_activation = intermediate_activation
        self.nuc_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            ignore_class=0, from_logits=False, reduction="none"
        )
        self.nuc_entropy = tf.keras.metrics.Mean()

        # layers used in model
        self.asv_encoder = ASVEncoder(
            max_bp,
            nuc_attention_heads,
            nuc_attention_layers,
            0.0,
            nuc_intermediate_size,
            intermediate_activation=self.intermediate_activation,
            name="asv_encoder",
        )
        self.nuc_logits = tf.keras.layers.Dense(
            6, use_bias=False, name="nuc_logits", dtype=tf.float32, activation="softmax"
        )

        self.asv_scale = tf.keras.layers.Dense(self.embedding_dim, use_bias=False)
        self.asv_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.asv_pos = tfm.nlp.layers.PositionEmbedding(self.token_limit + 5)

        self.sample_encoder = TransformerEncoder(
            num_layers=self.sample_attention_layers,
            num_attention_heads=self.sample_attention_heads,
            intermediate_size=self.sample_intermediate_size,
            activation=self.intermediate_activation,
            dropout_rate=self.dropout_rate,
        )
        self.sample_token = self.add_weight(
            "sample_token",
            [1, 1, self.embedding_dim],
            dtype=tf.float32,
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self._base_alpha = self.add_weight(
            name="base_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )

        self.linear_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)

    @masked_loss(sparse_cat=True)
    def _compute_nuc_loss(self, tokens: tf.Tensor, pred_tokens: tf.Tensor) -> tf.Tensor:
        return self.nuc_loss(tokens, pred_tokens)

    def _add_sample_token(self, tensor: tf.Tensor) -> tf.Tensor:
        # add <SAMPLE> token empbedding
        asv_shape = tf.shape(tensor)
        batch_len = asv_shape[0]
        emb_len = asv_shape[-1]
        sample_emb_shape = [1 for _ in tensor.get_shape().as_list()]
        sample_emb_shape[0] = batch_len
        sample_emb_shape[-1] = emb_len
        sample_token = tf.broadcast_to(self.sample_token, sample_emb_shape)
        embeddings = tf.concat([sample_token, tensor], axis=1)
        return embeddings

    def _split_asvs(self, embeddings):
        nuc_embeddings = embeddings[:, :, :-1, :]
        nucleotides = self.nuc_logits(nuc_embeddings)

        asv_embeddings = embeddings[:, :, 0, :]
        asv_embeddings = self.asv_norm(asv_embeddings)
        asv_embeddings = asv_embeddings + self.asv_pos(asv_embeddings)

        return asv_embeddings, nucleotides

    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        # need to cast inputs to int32 to avoid error
        # because keras converts all inputs
        # to float when calling build()
        asv_input = tf.cast(inputs, dtype=tf.int32)

        embeddings = self.asv_encoder(asv_input, training=training)
        embeddings = self.asv_scale(embeddings)
        asv_embeddings, nucleotides = self._split_asvs(embeddings)

        asv_mask = float_mask(tf.reduce_sum(inputs, axis=-1, keepdims=True))
        padded_asv_mask = tf.pad(asv_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)

        # padded embeddings are the skip connection
        # normal asv embeddings continue through next block
        padded_asv_embeddings = tf.pad(
            asv_embeddings, [[0, 0], [1, 0], [0, 0]], constant_values=0
        )

        sample_gated_embeddings = self._add_sample_token(asv_embeddings)
        sample_gated_embeddings = self.sample_encoder(
            sample_gated_embeddings, mask=padded_asv_mask, training=training
        )

        sample_embeddings = (
            padded_asv_embeddings + sample_gated_embeddings * self._base_alpha
        )
        return sample_embeddings, nucleotides

    def get_config(self):
        config = super(BaseSequenceEncoder, self).get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "max_bp": self.max_bp,
                "token_limit": self.token_limit,
                "sample_attention_heads": self.sample_attention_heads,
                "sample_attention_layers": self.sample_attention_layers,
                "sample_intermediate_size": self.sample_intermediate_size,
                "dropout_rate": self.dropout_rate,
                "nuc_attention_heads": self.nuc_attention_heads,
                "nuc_attention_layers": self.nuc_attention_layers,
                "nuc_intermediate_size": self.nuc_intermediate_size,
            }
        )
        return config
