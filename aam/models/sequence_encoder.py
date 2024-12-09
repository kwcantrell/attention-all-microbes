from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.losses import PairwiseLoss

# from aam.models.attention_pooling import AttentionPooling
from aam.models.base_sequence_encoder import BaseSequenceEncoder
from aam.models.multihead_attention_pooling import MultiHeadAttentionPooling
from aam.models.transformers import TransformerEncoder
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
            nuc_intermediate_size=256,
            intermediate_activation=self.intermediate_activation,
            is_16S=self.is_16S,
            vocab_size=self.vocab_size,
            add_token=self.add_token,
            nucleotide_encoder=nucleotide_encoder,
            name="base_encoder",
        )

        # self.attention_pooling = AttentionPooling()
        self.attention_pooling = MultiHeadAttentionPooling()

        self.encoder = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
            name="encoder",
        )

        if self.encoder_type == "combined":
            uni_out, faith_out, tax_out = self.output_dim
            self.uni_ff = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        self.embedding_dim, activation="gelu", dtype=tf.float32
                    ),
                    tf.keras.layers.Dense(uni_out, dtype=tf.float32),
                ]
            )
            self.faith_ff = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        self.embedding_dim, activation="gelu", dtype=tf.float32
                    ),
                    tf.keras.layers.Dense(faith_out, dtype=tf.float32),
                ]
            )
            self.tax_ff = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        self.embedding_dim, activation="gelu", dtype=tf.float32
                    ),
                    tf.keras.layers.Dense(tax_out, dtype=tf.float32),
                ]
            )
        elif self.encoder_type == "unifrac":
            self.encoder_ff = tf.keras.layers.Dense(self.output_dim, dtype=tf.float32)
        else:
            self.encoder_ff = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(self.embedding_dim * 4, activation="relu"),
                    tf.keras.layers.Dropout(0.1),
                    tf.keras.layers.Dense(self.embedding_dim * 4, activation="relu"),
                    tf.keras.layers.Dropout(0.1),
                    tf.keras.layers.Dense(self.output_dim, activation="softmax"),
                ]
            )

        self.gradient_accumulator = GradientAccumulator(self.accumulation_steps)
        self.loss_scaler = LossScaler(self.gradient_accumulator.accum_steps)

    @property
    def accumulation_steps(self):
        return self._accumulation_steps

    @accumulation_steps.setter
    def accumulation_steps(self, steps):
        self._accumulation_steps = steps
        self.gradient_accumulator = GradientAccumulator(self.accumulation_steps)

    def _get_encoder_loss(self):
        if self.encoder_type == "combined":
            self._unifrac_loss = PairwiseLoss(self.pairwise_loss_type)
            self._tax_loss = tf.keras.losses.CategoricalCrossentropy(reduction="none")
            self.encoder_loss = self._compute_combined_loss
            self.extract_encoder_pred = self._combined_embeddigns
        elif self.encoder_type == "unifrac":
            self._unifrac_loss = PairwiseLoss(self.pairwise_loss_type)
            self.encoder_loss = self._compute_unifrac_loss
            self.extract_encoder_pred = self._unifrac_embeddings
        elif self.encoder_type == "faith_pd":
            self._unifrac_loss = tf.keras.losses.MeanSquaredError(reduction="none")
            self.encoder_loss = self._compute_unifrac_loss
            self.extract_encoder_pred = self._unifrac_embeddings
        elif self.encoder_type == "taxonomy":
            self._tax_loss = tf.keras.losses.CategoricalCrossentropy(reduction="none")
            self.encoder_loss = self._compute_tax_loss
            self.extract_encoder_pred = self._taxonomy_embeddings
        else:
            raise Exception(f"invalid encoder encoder_type: {self.encoder_type}")

    def _combined_embeddigns(self, tensor, mask):
        if self.add_token:
            unifrac_pred = tensor[:, 0, :]
        else:
            mask = tf.cast(mask, dtype=tf.float32)
            unifrac_pred = tf.reduce_sum(tensor * mask, axis=1)
            unifrac_pred /= tf.reduce_sum(mask, axis=1)

        if self.add_token:
            faith_pred = tensor[:, 0, :]
        else:
            mask = tf.cast(mask, dtype=tf.float32)
            faith_pred = tf.reduce_sum(tensor * mask, axis=1)
            faith_pred /= tf.reduce_sum(mask, axis=1)

        tax_pred = tensor
        if self.add_token:
            tax_pred = tax_pred[:, 1:, :]

        return [
            self.uni_ff(unifrac_pred),
            self.faith_ff(faith_pred),
            self.tax_ff(tax_pred),
        ]

    def _unifrac_embeddings(self, tensor, mask=None, training=False):
        encoder_pred = self.attention_pooling(tensor, mask=mask, training=training)
        encoder_pred = self.encoder_ff(encoder_pred)
        return encoder_pred

    def _taxonomy_embeddings(self, tensor, mask=None, training=False):
        tax_pred = tensor
        tax_pred = self.encoder_ff(tax_pred, training=training)
        return tax_pred

    def _compute_combined_loss(self, y_true, preds):
        uni_true, faith_true, tax_true = y_true
        uni_pred, faith_pred, tax_pred = preds

        uni_loss = self._compute_unifrac_loss(uni_true, uni_pred)
        faith_loss = tf.reduce_mean(tf.square(faith_true - faith_pred))
        tax_loss = self._compute_tax_loss(tax_true, tax_pred)

        return [uni_loss, faith_loss, tax_loss]

    def _compute_tax_loss(
        self,
        tax_tokens: tf.Tensor,
        tax_pred: tf.Tensor,
    ) -> tf.Tensor:
        if isinstance(self.output_dim, (list, tuple)):
            out_dim = self.output_dim[-1]
        else:
            out_dim = self.output_dim
        y_true = tf.reshape(tax_tokens, [-1])
        y_pred = tf.reshape(tax_pred, [-1, out_dim])

        mask = float_mask(y_true) > 0
        y_true = tf.one_hot(y_true, depth=out_dim)

        # smooth labels
        y_true = y_true * 0.9 + 0.1

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        loss = tf.reduce_mean(self._tax_loss(y_true, y_pred))
        return loss

    def _compute_unifrac_loss(
        self,
        y_true: tf.Tensor,
        unifrac_embeddings: tf.Tensor,
    ) -> tf.Tensor:
        batch_size = tf.shape(y_true)[0]
        pairs = tf.linalg.band_part(
            tf.ones((batch_size, batch_size), dtype=tf.float32), 0, -1
        )
        pairs = tf.reduce_sum(pairs)

        loss = self._unifrac_loss(y_true, unifrac_embeddings)
        loss = tf.reduce_sum(loss / pairs)
        return loss / 2

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
        nuc_tokens, counts = model_inputs
        embeddings, encoder_embeddings, nuc_mask, nuc_pred = outputs

        nuc_tokens = nuc_tokens + self.base_encoder.asv_encoder.nucleotide_position
        nuc_tokens = tf.reshape(nuc_tokens, shape=[-1])
        nuc_mask = tf.reshape(nuc_mask, shape=[-1])
        nuc_tokens = nuc_tokens[nuc_mask]
        nuc_tokens = tf.one_hot(nuc_tokens, tf.shape(nuc_pred)[-1])
        nuc_loss = self.nuc_loss(nuc_tokens, nuc_pred)
        nuc_loss = tf.reduce_mean(nuc_loss)

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
        embeddings, encoder_embeddings, nuc_mask, nuc_pred = self(
            inputs, training=False
        )

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

        gradients = tape.gradient(
            loss,
            self.trainable_variables,
            # unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        # self.gradient_accumulator.apply_gradients(gradients)
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
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        # account for <SAMPLE> token
        count_mask = float_mask(counts, dtype=tf.int32)
        random_mask = None

        sample_embeddings, nuc_mask, nuc_pred = self.base_encoder(
            tokens, random_mask=random_mask, training=training
        )

        encoder_gated_embeddings = self.encoder(
            sample_embeddings, mask=count_mask, training=training
        )

        encoder_pred = self.extract_encoder_pred(
            encoder_gated_embeddings, count_mask, training=training
        )
        encoder_embeddings = encoder_gated_embeddings
        return encoder_embeddings, encoder_pred, nuc_mask, nuc_pred

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

        return self.base_encoder.asv_embeddings(tokens, training=training)

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
        nucleotide_encoder = tf.keras.saving.serialize_keras_object(
            self.base_encoder.asv_encoder
        )
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
                "nucleotide_encoder": nucleotide_encoder,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if hasattr(config, "nucleotide_encoder"):
            nucleotide_encoder = config["nucleotide_encoder"]
            if nucleotide_encoder is not None:
                config["nucleotide_encoder"] = None
        model = cls(**config)
        return model
