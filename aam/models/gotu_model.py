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
        dropout_rate: float = 0.0,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 512,
        intermediate_activation: str = "gelu",
        pairwise_loss_type="mse",
        gotu_count=None,
        base_model=None,
        **kwargs,
    ):
        super(GOTUModel, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.pairwise_loss_type = pairwise_loss_type
        self.gotu_count = gotu_count
        self.base_model = base_model
        self.base_model.trainable = False

        #  loss, early_rank_loss, late_rank_loss, wrong_token_loss
        self.loss_tracker = tf.keras.metrics.Mean()
        self.early_rank_tracker = tf.keras.metrics.Mean()
        self.late_rank_tracker = tf.keras.metrics.Mean()
        self.wrong_token_tracker = tf.keras.metrics.Mean()

        self.gotu_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")

        self.gotu_embedding_layer = tf.keras.layers.Embedding(
            self.gotu_count, self.embedding_dim
        )
        self.asv_norm_layer = tf.keras.layers.LayerNormalization(dtype=tf.float32)

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
        self.gotu_output = tf.keras.layers.Dense(self.gotu_count)

    def build(self, input_shape):
        if self.built:
            print("GOTU Model already built..")
            return
        super(GOTUModel, self).build(input_shape)

    def get_config(self):
        config = super(GOTUModel, self).get_config()
        config.update(
            {
                "dropout_rate": self.dropout_rate,
                "embedding_dim": self.embedding_dim,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "intermediate_activation": self.intermediate_activation,
                "gotu_count": self.gotu_count,
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        asv_encoder_config = config.pop("base_model")
        base_model = tf.keras.saving.deserialize_keras_object(asv_encoder_config)
        build_input_shape = config.pop("build_input_shape")
        input_shape = build_input_shape["input_shape"]
        model = cls(base_model=base_model, **config)
        model.build(input_shape)
        return model

    def _compute_loss(self, outputs):
        gotu_pred, gotu_tokens = outputs
        gotu_tokens = tf.cast(gotu_tokens, dtype=tf.float32)
        gotu_tokens = tf.pad(gotu_tokens, paddings=[[0, 0], [0, 1]], constant_values=2)

        tokens_per_sample = tf.shape(gotu_tokens)[-1]
        gotu_loss = self.gotu_loss(gotu_tokens, gotu_pred)

        gotu_valid_mask = tf.cast(gotu_tokens > 0, dtype=tf.float32)
        gotu_pred_tokens = tf.math.reduce_max(gotu_pred, axis=-1)
        gotu_pred_tokens = tf.expand_dims(gotu_pred_tokens, axis=-1)
        gotu_tokens = tf.expand_dims(gotu_tokens, axis=1)
        gotu_comparison_output = tf.cast(
            gotu_tokens == gotu_pred_tokens, dtype=tf.float32
        )
        correct_mask = (
            tf.math.reduce_max(gotu_comparison_output, axis=-1) * gotu_valid_mask
        )

        # loss for determining if token was called early or late
        true_rankings = tf.range(tokens_per_sample, dtype=tf.float32)
        rankings = (
            tf.cast(tf.argmax(gotu_comparison_output, axis=-1), dtype=tf.float32)
            - true_rankings
        )
        abs_ranking = tf.abs(rankings)
        rel_ranking = abs_ranking / tf.cast(tokens_per_sample, dtype=tf.float32)

        def _mean_loss(loss, loss_mask):
            loss = tf.math.divide_no_nan(
                tf.reduce_sum(loss, axis=-1, keepdims=True),
                tf.reduce_sum(loss_mask, axis=-1, keepdims=True),
            )
            loss = tf.reduce_mean(loss)

            return loss

        def _rankings_loss(early=True):
            if early:
                rank_mask = tf.cast(rankings < 0, dtype=tf.float32)
            else:
                rank_mask = tf.cast(rankings > 0, dtype=tf.float32)
            y_rank_scalar = rel_ranking * rank_mask * gotu_valid_mask
            rank_loss = gotu_loss * y_rank_scalar
            rank_loss = _mean_loss(rank_loss, rank_mask)
            return rank_loss

        early_rank_loss = _rankings_loss(early=True)
        late_rank_loss = _rankings_loss(early=False)

        # create wrong_token_loss
        wrong_mask = 1 - correct_mask
        wrong_token_loss = gotu_loss * wrong_mask
        wrong_token_loss = _mean_loss(wrong_token_loss, wrong_mask)

        loss = early_rank_loss + late_rank_loss + wrong_token_loss
        return loss, early_rank_loss, late_rank_loss, wrong_token_loss

    def batch_embeddings(self, embeddings, batch_indicies, counts, indices=None):
        emb_dim = tf.shape(embeddings)[-1]
        if indices is not None:
            embeddings = tf.gather(embeddings, indices)
        batch_shape = tf.reduce_max(batch_indicies[:, 0]) + 1
        max_unique = tf.reduce_max(batch_indicies[:, 1]) + 1
        batch_embeddings = tf.scatter_nd(
            batch_indicies, embeddings, shape=[batch_shape, max_unique, emb_dim]
        )
        counts = tf.scatter_nd(
            batch_indicies, counts, shape=[batch_shape, max_unique, 1]
        )
        batch_embeddings, counts = sort_using_counts(batch_embeddings, counts)
        return batch_embeddings, counts

    def train_step(self, data):
        inputs = data
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, early_rank_loss, late_rank_loss, wrong_token_loss = (
                self._compute_loss(outputs)
            )
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.early_rank_tracker.update_state(early_rank_loss)
        self.late_rank_tracker.update_state(late_rank_loss)
        self.wrong_token_tracker.update_state(wrong_token_loss)
        metrics = {
            "loss": self.loss_tracker.result(),
            "early_rank_loss": self.early_rank_tracker.result(),
            "late_rank_loss": self.late_rank_tracker.result(),
            "wrong_token_loss": self.wrong_token_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }
        return metrics

    def test_step(self, data):
        inputs = data
        outputs = self(inputs, training=False)
        loss, early_rank_loss, late_rank_loss, wrong_token_loss = self._compute_loss(
            outputs
        )

        self.loss_tracker.update_state(loss)
        self.early_rank_tracker.update_state(early_rank_loss)
        self.late_rank_tracker.update_state(late_rank_loss)
        self.wrong_token_tracker.update_state(wrong_token_loss)
        metrics = {
            "loss": self.loss_tracker.result(),
            "early_rank_loss": self.early_rank_tracker.result(),
            "late_rank_loss": self.late_rank_tracker.result(),
            "wrong_token_loss": self.wrong_token_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

        return metrics

    def predict_step(self, data):
        asv_inputs, gotu_inputs = data
        asv_x, _ = asv_inputs
        gotu_x, _ = gotu_inputs
        outputs = self((asv_x, gotu_x), training=True)

        return outputs[0]

    def cast_inputs(self, inputs):
        (
            asv_tokens,
            asv_batch_indices,
            asv_indicies,
            asv_counts,
            gotu_tokens,
            gotu_counts,
        ) = inputs
        asv_tokens = tf.cast(asv_tokens, dtype=tf.int32)
        asv_batch_indices = tf.cast(asv_batch_indices, dtype=tf.int32)
        asv_indicies = tf.cast(asv_indicies, dtype=tf.int32)
        asv_counts = tf.cast(asv_counts, dtype=tf.int32)
        gotu_tokens = tf.cast(gotu_tokens, dtype=tf.int32)
        gotu_counts = tf.cast(gotu_counts, dtype=tf.float32)

        # This is getting relative abundance per sample for gotus
        gotu_tokens_per_sample = tf.cast(gotu_tokens == 0, dtype=tf.float32)
        gotu_tokens_per_sample = tf.reduce_sum(
            gotu_tokens_per_sample, axis=-1, keepdims=True
        )
        gotu_counts_per_sample = (
            tf.reduce_sum(gotu_counts, axis=-1, keepdims=True) / gotu_tokens_per_sample
        )

        gotu_counts = gotu_counts / gotu_counts_per_sample
        return (
            asv_tokens,
            asv_batch_indices,
            asv_indicies,
            asv_counts,
            gotu_tokens,
            gotu_counts,
        )

    def extract_gotu_embeddings(
        self, gotu_tokens, gotu_counts, asv_embeddings, asv_mask, training=False
    ):
        gotu_pad_mask = tf.cast(gotu_tokens == 0, dtype=tf.int32) * 2
        gotu_tokens = gotu_tokens + gotu_pad_mask
        gotu_embeddings = self.gotu_embedding_layer(gotu_tokens)

        gotu_mask = tf.cast(gotu_counts > 0, dtype=self.compute_dtype)
        gotu_pred = self.gotu_decoder(
            (asv_embeddings, gotu_embeddings), asv_mask, gotu_mask, training=training
        )
        gotu_pred = self.gotu_output(gotu_pred)
        return gotu_pred

    def call(
        self,
        inputs,
        add_start_token=True,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        (
            asv_tokens,
            asv_batch_indices,
            asv_indicies,
            asv_counts,
            gotu_tokens,
            gotu_counts,
        ) = self.cast_inputs(inputs)
        asv_embeddings, asv_counts = self.base_model.extract_asv_embeddings(
            (asv_tokens, asv_batch_indices, asv_indicies, asv_counts),
            batch_embeddings=True,
            sort_counts=True,
        )
        asv_embeddings = self.asv_norm_layer(asv_embeddings)
        asv_embeddings = tf.cast(asv_embeddings, dtype=self.compute_dtype)
        asv_mask = tf.cast(asv_counts > 0, dtype=self.compute_dtype)

        gotu_tokens_with_start = tf.expand_dims(gotu_tokens, axis=-1)
        gotu_counts = tf.expand_dims(gotu_counts, axis=-1)

        gotu_tokens_with_start, gotu_counts = sort_using_counts(
            gotu_tokens_with_start, gotu_counts
        )
        if add_start_token:
            gotu_tokens_with_start = tf.pad(
                gotu_tokens_with_start,
                paddings=[[0, 0], [1, 0], [0, 0]],
                constant_values=1,
            )
            gotu_counts = tf.pad(
                gotu_counts, paddings=[[0, 0], [1, 0], [0, 0]], constant_values=1
            )
        gotu_tokens_with_start = tf.squeeze(gotu_tokens_with_start, axis=-1)
        gotu_pred = self.extract_gotu_embeddings(
            gotu_tokens_with_start, gotu_counts, asv_embeddings, asv_mask, training
        )
        gotu_pred = self._softmax(gotu_pred)

        return gotu_pred, gotu_tokens
