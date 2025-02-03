from __future__ import annotations

import inspect

import tensorflow as tf
import tensorflow_models as tfm

from aam.losses import PairwiseLoss
from aam.models.transformer_decoder import TransformerDecoder
from aam.models.unifrac_encoder import UnifracEncoder
from aam.models.utils import sort_using_counts, to_batch


@tf.keras.saving.register_keras_serializable(package="GOTUModel")
class GOTUModel(tf.keras.Model):
    def __init__(
        self,
        output_dim: int,
        token_limit: int,
        encoder_type: str = "unifrac",
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
        gotu_count=None,
        max_gotu=2048,
        asv_embedding_layer=None,
        freeze_base_weights=False,
        bert_training=False,
        **kwargs,
    ):
        super(GOTUModel, self).__init__(**kwargs)
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
        self.gotu_count = gotu_count
        self.max_gotu = max_gotu
        self.asv_embedding_layer = asv_embedding_layer
        self.asv_embedding_layer.trainable = False
        self.freeze_base_weights = freeze_base_weights
        self.bert_training = bert_training

        self.encoder_tracker = tf.keras.metrics.Mean()
        self.loss_tracker = tf.keras.metrics.Mean()
        self.gotu_tracker = tf.keras.metrics.Mean()
        self.nuc_tracker = tf.keras.metrics.Mean()

        self.nuc_loss = tf.keras.losses.CategoricalCrossentropy(reduction="none")
        self.encoder_loss = PairwiseLoss(self.pairwise_loss_type, reduction="none")
        self.gotu_loss = tf.keras.losses.CategoricalCrossentropy(reduction="none")

        self.gotu_embedding_layer = tf.keras.layers.Embedding(self.gotu_count + 2, self.embedding_dim)

        self.gotu_decoder = TransformerDecoder(
            num_attention_heads=self.attention_heads,
            num_layers=self.attention_layers,
            intermediate_size=self.intermediate_size,
            activation=self.intermediate_activation,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.dropout_rate,
            use_linear_bias=True,
        )
        self._softmax = tf.keras.layers.Activation("softmax", dtype=tf.float32)
        self.gotu_output = tf.keras.layers.Dense(self.gotu_count + 2)

    def get_config(self):
        config = super(GOTUModel, self).get_config()
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
                "gotu_count": self.gotu_count,
                "max_gotu": self.max_gotu,
                "freeze_base_weights": self.freeze_base_weights,
                "asv_embedding_layer": self.asv_embedding_layer.get_config(),
                "bert_training": self.bert_training,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        asv_embedding_layer_config = config.pop("asv_embedding_layer")
        asv_embedding_layer = UnifracEncoder.from_config(asv_embedding_layer_config)
        model = cls(asv_embedding_layer=asv_embedding_layer, **config)
        return model

    def _compute_loss(self, asv_inputs, gotu_inputs, asv_unifrac_dist, outputs):
        # asv_batch_counts, asv_tokens, asv_indicies, _ = asv_inputs
        # gotu_batch_counts, gotu_tokens, gotu_counts = gotu_inputs
        # gotu_pred, unifrac_pred = outputs
        # nuc_loss = self.asv_embedding_layer.losses

        # # compute unifrac loss
        # unifrac_loss = self.asv_embedding_layer._compute_unifrac_loss(
        #     asv_unifrac_dist, unifrac_pred
        # )

        # # compute decoder loss
        # gotu_tokens = to_batch(gotu_tokens, gotu_batch_counts)
        # gotu_counts = to_batch(gotu_counts, gotu_batch_counts)
        # gotu_tokens, gotu_counts = sort_using_counts(gotu_tokens, gotu_counts)
        # gotu_tokens = tf.pad(
        #     gotu_tokens,
        #     [[0, 0], [0, 1], [0, 0]],
        #     constant_values=2,
        # )

        # gotu_mask = tf.reshape(gotu_tokens > 1, shape=[-1])
        # gotu_tokens = tf.squeeze(gotu_tokens, axis=-1)
        # gotu_tokens = tf.one_hot(gotu_tokens, tf.shape(gotu_pred)[-1])
        # gotu_tokens = tf.reshape(gotu_tokens, shape=[-1, tf.shape(gotu_pred)[-1]])[
        #     gotu_mask
        # ]
        # gotu_pred = tf.reshape(gotu_pred, shape=[-1, tf.shape(gotu_pred)[-1]])[
        #     gotu_mask
        # ]
        # gotu_loss = self.gotu_loss(gotu_tokens, gotu_pred)
        # gotu_loss = tf.reduce_mean(gotu_loss)

        # loss = gotu_loss
        # if not self.freeze_base_weights:
        #     loss += unifrac_loss + nuc_loss
        loss = 0
        gotu_loss = 0
        nuc_loss = 0
        unifrac_loss = 0
        return loss, gotu_loss, nuc_loss, unifrac_loss

    def extract_data(self, data):
        asv_data, gotu_data = data
        asv_inputs, asv_targets = asv_data
        gotu_inputs, gotu_targets = gotu_data
        gotu_tokens, gotu_batch_indices, gotu_indices, gotu_counts = gotu_inputs
        gotu_batch_tokens, counts = self.batch_embeddings(gotu_tokens, gotu_batch_indices, gotu_counts, gotu_indices)

        return (asv_inputs, (gotu_batch_tokens, counts)), (asv_targets, gotu_targets)

    def batch_embeddings(self, embeddings, batch_indicies, counts, indices=None):
        emb_dim = tf.shape(embeddings)[-1]
        if indices is not None:
            embeddings = tf.gather(embeddings, indices)
        batch_shape = tf.reduce_max(batch_indicies[:, 0]) + 1
        max_unique = tf.reduce_max(batch_indicies[:, 1]) + 1
        batch_embeddings = tf.scatter_nd(batch_indicies, embeddings, shape=[batch_shape, max_unique, emb_dim])
        counts = tf.scatter_nd(batch_indicies, counts, shape=[batch_shape, max_unique, 1])
        batch_embeddings, counts = sort_using_counts(batch_embeddings, counts)
        return batch_embeddings, counts

    def train_step(self, data):
        inputs, targets = self.extract_data(data)
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, gotu_loss, nuc_loss, encoder_loss = self._compute_loss(targets, outputs)
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.gotu_tracker.update_state(gotu_loss)
        metrics = {
            "loss": self.loss_tracker.result(),
            "gotu_loss": self.gotu_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }
        if not self.freeze_base_weights:
            self.encoder_tracker.update_state(encoder_loss)
            self.nuc_tracker.update_state(nuc_loss)
            metrics.update(
                {
                    "encoder_loss": self.encoder_tracker.result(),
                    "nuc_loss": self.nuc_tracker.result(),
                }
            )
        return metrics

    def test_step(self, data):
        inputs, targets = self.extract_data(data)
        outputs = self(inputs, training=False)
        loss, gotu_loss, nuc_loss, encoder_loss = self._compute_loss(targets, outputs)

        self.loss_tracker.update_state(loss)
        self.gotu_tracker.update_state(gotu_loss)
        metrics = {
            "loss": self.loss_tracker.result(),
            "gotu_loss": self.gotu_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }
        if not self.freeze_base_weights:
            self.encoder_tracker.update_state(encoder_loss)
            self.nuc_tracker.update_state(nuc_loss)
            metrics.update(
                {
                    "encoder_loss": self.encoder_tracker.result(),
                    "nuc_loss": self.nuc_tracker.result(),
                }
            )
        return metrics

    def predict_step(self, data):
        asv_inputs, gotu_inputs = data
        asv_x, _ = asv_inputs
        gotu_x, _ = gotu_inputs
        outputs = self((asv_x, gotu_x), training=True)

        return outputs[0]

    def call(
        self,
        inputs,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        asv_inputs, gotu_inputs = inputs
        gotu_tokens, gotu_counts = gotu_inputs
        gotu_mask = tf.cast(gotu_counts > 0, dtype=self.compute_dtype)

        asv_embeddings, asv_counts = self.asv_embedding_layer(
            asv_inputs,
            return_asv_embeddings=True,
            training=False,
        )
        asv_mask = tf.cast(asv_counts > 0, dtype=self.compute_dtype)
        gotu_embeddings = self.gotu_embedding_layer(gotu_tokens)
        gotu_pred = self.gotu_decoder(asv_embeddings, gotu_embeddings, asv_mask, gotu_mask, training=training)
        gotu_pred = self.gotu_output(gotu_pred)
        gotu_pred = self._softmax(gotu_pred)

        return gotu_pred
