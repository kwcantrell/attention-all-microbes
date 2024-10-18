from __future__ import annotations

import tensorflow as tf
import tensorflow_models as tfm

from aam.layers import (
    ASVEncoder,
)
from aam.utils import float_mask, masked_loss


@tf.keras.saving.register_keras_serializable(package="BaseSequenceEncoder")
class BaseSequenceEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim: int,
        max_bp: int,
        sample_attention_heads: int,
        sample_attention_layers: int,
        sample_intermediate_size: int,
        dropout_rate: float,
        nuc_attention_heads: int = 2,
        nuc_attention_layers: int = 4,
        nuc_intermediate_size: int = 1024,
        **kwargs,
    ):
        super(BaseSequenceEncoder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.max_bp = max_bp
        self.sample_attention_heads = sample_attention_heads
        self.sample_attention_layers = sample_attention_layers
        self.sample_intermediate_size = sample_intermediate_size
        self.dropout_rate = dropout_rate
        self.nuc_attention_heads = nuc_attention_heads
        self.nuc_attention_layers = nuc_attention_layers
        self.nuc_intermediate_size = nuc_intermediate_size
        self.nuc_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            ignore_class=0, from_logits=False, reduction="none"
        )
        self.nuc_entropy = tf.keras.metrics.Mean()

        # layers used in model
        self.asv_encoder = ASVEncoder(
            max_bp,
            nuc_attention_heads,
            nuc_attention_layers,
            dropout_rate,
            nuc_intermediate_size,
            name="asv_encoder",
        )
        self.nuc_logits = tf.keras.layers.Dense(
            6, name="nuc_logits", dtype=tf.float32, activation="softmax"
        )

        self.asv_scale = tf.keras.layers.Dense(embedding_dim, activation="relu")
        self.asv_norm = tf.keras.layers.LayerNormalization(epsilon=0.000001)
        self.asv_pos = tfm.nlp.layers.PositionEmbedding(515)

        self.sample_encoder = tfm.nlp.models.TransformerEncoder(
            num_layers=self.sample_attention_layers,
            num_attention_heads=self.sample_attention_heads,
            intermediate_size=self.sample_intermediate_size,
            norm_first=True,
            activation="relu",
            dropout_rate=self.dropout_rate,
        )
        self.sample_token = self.add_weight(
            "sample_token",
            [1, 1, self.embedding_dim],
            dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )

        self.linear_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)

    @masked_loss(sparse_cat=True)
    def _compute_nuc_loss(self, tokens: tf.Tensor, pred_tokens: tf.Tensor) -> tf.Tensor:
        return self.nuc_loss(tokens, pred_tokens)

    def _add_sample_token(self, tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        # add <SAMPLE> token empbedding
        asv_shape = tf.shape(tensor)
        batch_len = asv_shape[0]
        emb_len = asv_shape[-1]
        sample_emb_shape = [1 for _ in tensor.get_shape().as_list()]
        sample_emb_shape[0] = batch_len
        sample_emb_shape[-1] = emb_len
        sample_token = tf.broadcast_to(self.sample_token, sample_emb_shape)
        embeddings = tf.concat([sample_token, tensor], axis=1)
        mask = tf.pad(mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        return embeddings, mask

    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        # need to cast inputs to int32 to avoid error
        # because keras converts all inputs
        # to float when calling build()
        asv_input = tf.cast(inputs, dtype=tf.int32)
        embeddings = self.asv_encoder(
            asv_input,
            training=training,
        )
        nuc_embeddings = embeddings[:, :, :-1, :]
        nucleotides = self.nuc_logits(nuc_embeddings)

        attention_mask = float_mask(tf.reduce_sum(inputs, axis=-1, keepdims=True))

        asv_embeddings = self.asv_scale(embeddings[:, :, -1, :])
        sample_embeddings, attention_mask = self._add_sample_token(
            asv_embeddings, attention_mask
        )
        sample_embeddings = self.asv_norm(sample_embeddings)
        sample_embeddings = sample_embeddings + self.asv_pos(sample_embeddings)

        attention_mask = tf.matmul(attention_mask, attention_mask, transpose_b=True)
        sample_embeddings = self.sample_encoder(
            sample_embeddings, attention_mask=attention_mask, training=training
        )

        return sample_embeddings, nucleotides

    def build(self, input_shape=None):
        super(BaseSequenceEncoder, self).build(
            tf.TensorShape([None, None, self.max_bp])
        )

    def get_config(self):
        config = super(BaseSequenceEncoder, self).get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "max_bp": self.max_bp,
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
