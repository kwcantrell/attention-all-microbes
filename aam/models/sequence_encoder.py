from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.losses import PairwiseLoss

# from aam.models.attention_pooling import AttentionPooling
from aam.models.base_sequence_encoder import BaseSequenceEncoder
from aam.models.multihead_attention_pooling import MultiHeadAttentionPooling
from aam.models.transformers import TransformerEncoder
from aam.models.utils import sort_using_counts, to_batch
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler
from aam.utils import float_mask


@tf.keras.saving.register_keras_serializable(package="SequenceEncoder")
class SequenceEncoder(tf.keras.Model):
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
        normalize_outputs=True,
        **kwargs,
    ):
        super(SequenceEncoder, self).__init__(**kwargs)
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

        self._get_encoder_loss()
        self.loss_tracker = tf.keras.metrics.Mean()
        self.encoder_tracker = tf.keras.metrics.Mean()

        self.nuc_loss = tf.keras.losses.CategoricalCrossentropy(reduction="none")
        self.nuc_tracker = tf.keras.metrics.Mean()

        # layers used in model
        self.base_encoder = BaseSequenceEncoder(
            self.embedding_dim,
            self.max_bp,
            self.token_limit,
            sample_attention_heads=self.attention_heads,
            sample_attention_layers=self.attention_layers,
            sample_intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            nuc_attention_heads=4,
            nuc_attention_layers=4,
            nuc_intermediate_size=512,
            intermediate_activation=self.intermediate_activation,
            is_16S=self.is_16S,
            vocab_size=self.vocab_size,
            add_token=self.add_token,
            nucleotide_encoder=nucleotide_encoder,
            normalize_outputs=self.normalize_outputs,
            name="base_encoder",
        )

        self.attention_pooling = MultiHeadAttentionPooling(self.normalize_outputs)
        self.encoder = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
            normalize_outputs=self.normalize_outputs,
            name="encoder",
        )

        self.encoder_ff = tf.keras.layers.Dense(self.output_dim, dtype=tf.float32)

        self.gradient_accumulator = GradientAccumulator(self.accumulation_steps)
        self.loss_scaler = LossScaler(self.gradient_accumulator.accum_steps)

        # asv_tokens, asv_indicies, asv_counts = [[None, self.max_bp], [None], [None, 1]]
        # self.inputs = [
        #     tf.keras.Input(shape=[], batch_size=None),
        #     tf.keras.Input(asv_tokens),
        #     tf.keras.Input(asv_indicies),
        #     tf.keras.Input(asv_counts),
        # ]
        # self.outputs = self.call(self.inputs)

    @property
    def accumulation_steps(self):
        return self._accumulation_steps

    @accumulation_steps.setter
    def accumulation_steps(self, steps):
        self._accumulation_steps = steps
        self.gradient_accumulator = GradientAccumulator(self.accumulation_steps)

    def _get_encoder_loss(self):
        self._unifrac_loss = PairwiseLoss(self.pairwise_loss_type)
        self.encoder_loss = self._compute_unifrac_loss
        self.extract_encoder_pred = self._unifrac_embeddings

    def _unifrac_embeddings(self, tensor, mask=None, training=False):
        encoder_pred = self.attention_pooling(tensor, mask=mask, training=training)
        encoder_pred = self.encoder_ff(encoder_pred)
        return encoder_pred

    def _compute_unifrac_loss(
        self,
        y_true: tf.Tensor,
        unifrac_embeddings: tf.Tensor,
    ) -> tf.Tensor:
        # loss = self._unifrac_loss(y_true, unifrac_embeddings)
        # loss = tf.reduce_max(loss, axis=-1)
        # return tf.reduce_mean(loss)
        batch_size = tf.shape(y_true)[0]
        pairs = tf.linalg.band_part(
            tf.ones((batch_size, batch_size), dtype=tf.float32), 0, -1
        )
        pairs = tf.reduce_sum(pairs) - tf.cast(batch_size, dtype=tf.float32)

        loss = self._unifrac_loss(y_true, unifrac_embeddings)
        loss = tf.reduce_sum(loss / 2)
        return loss / pairs

    def _compute_encoder_loss(
        self,
        y_true: tf.Tensor,
        encoder_embeddings: tf.Tensor,
    ) -> tf.Tensor:
        loss = self.encoder_loss(y_true, encoder_embeddings)
        if self.encoder_type != "combined":
            return tf.reduce_mean(loss)
        else:
            return loss

    def _compute_loss(
        self,
        model_inputs: tuple[tf.Tensor, tf.Tensor],
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_counts, nuc_tokens, indicies, counts = model_inputs
        embeddings, encoder_embeddings = outputs

        nuc_loss = tf.reduce_sum(self.base_encoder.losses)

        encoder_loss = self._compute_encoder_loss(y_true, encoder_embeddings)
        loss = nuc_loss + encoder_loss
        return loss, nuc_loss, encoder_loss

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        embeddings, encoder_embeddings = self.call(inputs, training=False)

        return encoder_embeddings, y

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        if not self.gradient_accumulator.built:
            self.gradient_accumulator.build(self.optimizer, self)

        inputs, y = data
        y_target, encoder_target = y
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, nuc_loss, encoder_loss = self._compute_loss(
                inputs, encoder_target, outputs
            )

            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.encoder_tracker.update_state(encoder_loss)
        self.nuc_tracker.update_state(nuc_loss)

        return {
            "loss": self.loss_tracker.result(),
            "encoder_loss": self.encoder_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def test_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        y_target, encoder_target = y
        outputs = self(inputs, training=False)
        loss, nuc_loss, encoder_loss = self._compute_loss(
            inputs, encoder_target, outputs
        )
        self.loss_tracker.update_state(loss)
        self.encoder_tracker.update_state(encoder_loss)
        self.nuc_tracker.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "encoder_loss": self.encoder_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def call(
        self,
        inputs,
        include_bert_random_mask=True,
        return_counts=False,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_counts, tokens, indicies, counts = inputs

        sample_embeddings = self.base_encoder(
            tokens, include_bert_random_mask=include_bert_random_mask, training=training
        )
        sample_embeddings = tf.gather(
            sample_embeddings, tf.cast(indicies, dtype=tf.int32)
        )
        sample_embeddings = to_batch(sample_embeddings, batch_counts)

        counts = to_batch(counts, batch_counts)
        sample_embeddings, counts = sort_using_counts(sample_embeddings, counts)
        count_mask = float_mask(counts, dtype=self.compute_dtype)

        encoder_gated_embeddings = self.encoder(
            sample_embeddings, mask=count_mask, training=training
        )

        encoder_pred = self.extract_encoder_pred(
            encoder_gated_embeddings, count_mask, training=training
        )
        encoder_embeddings = encoder_gated_embeddings
        if not return_counts:
            return encoder_embeddings, encoder_pred
        else:
            return encoder_embeddings, counts, encoder_pred

    def base_embeddings(
        self, inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        sample_embeddings = self.base_encoder.base_embeddings(tokens)

        # account for <SAMPLE> token
        count_mask = float_mask(counts, dtype=tf.int32)
        count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        count_attention_mask = count_mask

        unifrac_gated_embeddings = self.unifrac_encoder(
            sample_embeddings, mask=count_attention_mask
        )
        unifrac_pred = unifrac_gated_embeddings[:, 0, :]
        unifrac_pred = self.unifrac_ff(unifrac_pred)

        unifrac_embeddings = (
            sample_embeddings + unifrac_gated_embeddings * self._unifrac_alpha
        )

        return unifrac_embeddings

    def asv_embeddings(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens = inputs
        sample_embeddings = self.base_encoder(tokens, training=False)
        return sample_embeddings

    def asv_gradient(
        self, inputs: tuple[tf.Tensor, tf.Tensor], asv_embeddings
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        sample_embeddings = self.base_encoder.asv_gradient(tokens, asv_embeddings)

        # account for <SAMPLE> token
        count_mask = float_mask(counts, dtype=tf.int32)
        count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        count_attention_mask = count_mask

        unifrac_gated_embeddings = self.unifrac_encoder(
            sample_embeddings, mask=count_attention_mask
        )
        unifrac_pred = unifrac_gated_embeddings[:, 0, :]
        unifrac_pred = self.unifrac_ff(unifrac_pred)

        unifrac_embeddings = (
            sample_embeddings + unifrac_gated_embeddings * self._unifrac_alpha
        )

        return unifrac_embeddings

    def set_nucleotide_encoder(self, nucleotide_encoder):
        self.base_encoder.asv_encoder = nucleotide_encoder

    def get_config(self):
        config = super(SequenceEncoder, self).get_config()
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
            }
        )
        return config
