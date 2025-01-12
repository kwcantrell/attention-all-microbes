from __future__ import annotations

import tensorflow as tf

from aam.layers import ASVEncoder

# from aam.models.attention_pooling import AttentionPooling
from aam.models.multihead_attention_pooling import MultiHeadAttentionPooling
from aam.models.transformers import TransformerEncoder
from aam.utils import float_mask


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
        is_16S: bool = True,
        vocab_size: int = 6,
        add_token: bool = True,
        nucleotide_encoder=None,
        normalize_outputs=True,
        use_residual_connections=False,
        use_residual_pool=None,
        asv_encoder=None,
        regularize_embeddings=False,
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
        self.is_16S = is_16S
        self.vocab_size = vocab_size
        self.add_token = add_token
        self.nucleotide_encoder = nucleotide_encoder
        self.normalize_outputs = normalize_outputs
        self.use_residual_connections = use_residual_connections
        self.asv_encoder = asv_encoder
        self.regularize_embeddings = regularize_embeddings
        if use_residual_pool is None:
            use_residual_pool = use_residual_connections
        self.use_residual_pool = use_residual_pool

    def build(self, input_shape):
        # layers used in model
        if self.is_16S and self.asv_encoder is None:
            self.asv_encoder = ASVEncoder(
                self.max_bp,
                self.nuc_attention_heads,
                self.nuc_attention_layers,
                self.dropout_rate,
                self.nuc_intermediate_size,
                intermediate_activation=self.intermediate_activation,
                add_token=self.add_token,
                embedding_dim=self.embedding_dim,
                normalize_outputs=self.normalize_outputs,
                use_residual_connections=self.use_residual_connections,
                regularize_embeddings=self.regularize_embeddings,
                name="asv_encoder",
            )
        elif not self.is_16S:
            self.asv_embeddings = tf.keras.layers.Embedding(
                self.vocab_size,
                output_dim=self.embedding_dim,
                embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=self.embedding_dim**0.5),
            )
            self.asv_encoder = TransformerEncoder(
                num_layers=self.sample_attention_layers,
                num_attention_heads=self.sample_attention_heads,
                intermediate_size=self.sample_intermediate_size,
                activation=self.intermediate_activation,
                dropout_rate=self.dropout_rate,
                normalize_outputs=self.normalize_outputs,
            )

        self.attention_pool = MultiHeadAttentionPooling(
            self.normalize_outputs, num_heads=self.nuc_attention_heads, use_residual_connections=self.use_residual_pool
        )
        super(BaseSequenceEncoder, self).build(input_shape)

    def _split_asvs(self, embeddings, training):
        if self.is_16S:
            embeddings = self.attention_pool(embeddings, training=training)
        else:
            embeddings = embeddings[:, :, 0, :]

        return embeddings

    def call(self, inputs: tf.Tensor, include_bert_random_mask=True, training: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
        embeddings = self.asv_encoder(inputs, include_bert_random_mask=include_bert_random_mask, training=training)
        asv_embeddings = self._split_asvs(embeddings, training=training)
        print("BaseSequenceEncoder exit...")
        return asv_embeddings

    # def base_embeddings(
    #     self, inputs: tf.Tensor, training: bool = False
    # ) -> tuple[tf.Tensor, tf.Tensor]:
    #     # need to cast inputs to int32 to avoid error
    #     # because keras converts all inputs
    #     # to float when calling build()
    #     asv_input = tf.cast(inputs, dtype=tf.int32)

    #     embeddings = self.asv_encoder(asv_input, training=training)
    #     embeddings = self.asv_scale(embeddings)
    #     asv_embeddings, nucleotides = self._split_asvs(embeddings)

    #     asv_mask = float_mask(tf.reduce_sum(inputs, axis=-1, keepdims=True))
    #     padded_asv_mask = tf.pad(asv_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)

    #     # padded embeddings are the skip connection
    #     # normal asv embeddings continue through next block
    #     padded_asv_embeddings = tf.pad(
    #         asv_embeddings, [[0, 0], [1, 0], [0, 0]], constant_values=0
    #     )

    #     sample_gated_embeddings = self._add_sample_token(asv_embeddings)
    #     sample_gated_embeddings = self.sample_encoder(
    #         sample_gated_embeddings, mask=padded_asv_mask, training=training
    #     )

    #     sample_embeddings = (
    #         padded_asv_embeddings + sample_gated_embeddings * self._base_alpha
    #     )
    #     return sample_embeddings

    def asv_embeddings(self, inputs: tf.Tensor, training: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
        # need to cast inputs to int32 to avoid error
        # because keras converts all inputs
        # to float when calling build()
        asv_input = tf.cast(inputs, dtype=tf.int32)
        # boolean mask used to select non-pad tokens
        mask = tf.reduce_sum(asv_input, axis=-1) > 0  # shape [B, A]
        mask = tf.reshape(mask, shape=[-1])  # shape [B * A]

        # create indices for non-pad locations
        indices = tf.where(mask)

        embeddings, _, _ = self.asv_encoder(asv_input, training=training)
        return self._split_asvs(embeddings, mask, indices, training=training)

    def asv_gradient(self, inputs: tf.Tensor, asv_embeddings) -> tuple[tf.Tensor, tf.Tensor]:
        asv_mask = float_mask(tf.reduce_sum(inputs, axis=-1, keepdims=True))

        if self.add_token:
            asv_mask = tf.pad(asv_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
            sample_embeddings = self._add_sample_token(asv_embeddings)
        else:
            sample_embeddings = asv_embeddings

        sample_gated_embeddings = self.sample_encoder(sample_embeddings, mask=asv_mask, training=False)
        sample_embeddings = sample_embeddings + sample_gated_embeddings
        return sample_embeddings

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
                "intermediate_activation": self.intermediate_activation,
                "is_16S": self.is_16S,
                "vocab_size": self.vocab_size,
                "add_token": self.add_token,
                "normalize_outputs": self.normalize_outputs,
                "use_residual_connections": self.use_residual_connections,
                "use_residual_pool": self.use_residual_pool,
                "asv_encoder": tf.keras.saving.serialize_keras_object(self.asv_encoder),
                "regularize_embeddings": self.regularize_embeddings,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["asv_encoder"] = tf.keras.saving.deserialize_keras_object(config["asv_encoder"])
        model = cls(**config)
        return model
