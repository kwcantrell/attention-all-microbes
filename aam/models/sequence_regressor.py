from __future__ import annotations

from typing import Optional, Union

import tensorflow as tf
import tensorflow_models as tfm

# from aam.models.attention_pooling import AttentionPooling
from aam.models.multihead_attention_pooling import MultiHeadAttentionPooling

# from aam.models.unifrac_encoder import UniFracEncoder
from aam.models.sequence_encoder import SequenceEncoder
from aam.models.transformers import TransformerEncoder
from aam.models.utils import to_batch
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler
from aam.utils import create_random_mask, float_mask


@tf.keras.saving.register_keras_serializable(package="SequenceRegressor")
class SequenceRegressor(tf.keras.Model):
    def __init__(
        self,
        token_limit: int,
        base_output_dim: Optional[int] = None,
        shift: float = 0.0,
        scale: float = 1.0,
        dropout_rate: float = 0.0,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 1024,
        intermediate_activation: str = "relu",
        base_model: str = "unifrac",
        freeze_base: bool = False,
        penalty: float = 1.0,
        nuc_penalty: float = 1.0,
        max_bp: int = 150,
        is_16S: bool = True,
        vocab_size: int = 6,
        out_dim: int = 1,
        classifier: bool = False,
        add_token: bool = True,
        class_weights: list = None,
        asv_dropout_rate: float = 0.0,
        accumulation_steps: int = 1,
        scale_losses=False,
        **kwargs,
    ):
        super(SequenceRegressor, self).__init__(**kwargs)
        self.token_limit = token_limit
        self.base_output_dim = base_output_dim
        self.shift = shift
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.freeze_base = freeze_base
        self.penalty = penalty
        self.nuc_penalty = nuc_penalty
        self.max_bp = max_bp
        self.is_16S = is_16S
        self.vocab_size = vocab_size
        self.out_dim = out_dim
        self.classifier = classifier
        self.add_token = add_token
        self.class_weights = class_weights
        self.asv_dropout_rate = asv_dropout_rate
        self.accumulation_steps = accumulation_steps
        self.scale_losses = scale_losses
        self.loss_tracker = tf.keras.metrics.Mean()

        # layers used in model
        self.combined_base = False
        if isinstance(base_model, str):
            self.base_model = SequenceEncoder(
                output_dim=self.base_output_dim,
                token_limit=self.token_limit,
                encoder_type=base_model,
                dropout_rate=self.dropout_rate,
                embedding_dim=self.embedding_dim,
                attention_heads=self.attention_heads,
                attention_layers=self.attention_layers,
                intermediate_size=self.intermediate_size,
                intermediate_activation=self.intermediate_activation,
                max_bp=self.max_bp,
                is_16S=self.is_16S,
                vocab_size=self.vocab_size,
                add_token=self.add_token,
                asv_dropout_rate=self.asv_dropout_rate,
                accumulation_steps=self.accumulation_steps,
            )
        else:
            self.base_model = base_model

        self.base_losses = {"base_loss": self.base_model._compute_encoder_loss}
        self.base_metrics = {
            "base_loss": ["encoder_loss", self.base_model.encoder_tracker]
        }

        if self.freeze_base:
            print("Freezing base model...")
            self.base_model.trainable = False

        self.count_encoder = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
        )
        self.count_pos = tfm.nlp.layers.PositionEmbedding(
            self.token_limit + 5, initializer="zeros"
        )
        self.count_out = tf.keras.layers.Dense(1, dtype=tf.float32)
        self._rezero_a = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )
        self.count_loss = tf.keras.losses.MeanSquaredError(reduction="none")
        self.count_tracker = tf.keras.metrics.Mean()

        self.target_encoder = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
        )

        self.target_tracker = tf.keras.metrics.Mean()
        if not self.classifier:
            self.metric_tracker = tf.keras.metrics.MeanAbsoluteError()
            self.metric_string = "mae"
        else:
            self.metric_tracker = tf.keras.metrics.SparseCategoricalAccuracy()
            self.metric_string = "accuracy"

        self.attention_pooling = MultiHeadAttentionPooling()
        self.target_ff = tf.keras.layers.Dense(self.out_dim, dtype=tf.float32)

        self.loss_metrics = sorted(
            ["loss", "target_loss", "count_mse", self.metric_string]
        )
        self.gradient_accumulator = GradientAccumulator(self.accumulation_steps)
        self.loss_scaler = LossScaler(self.gradient_accumulator.accum_steps)

    def evaluate_metric(self, dataset, metric, **kwargs):
        metric_index = self.loss_metrics.index(metric)
        evaluated_metrics = super(SequenceRegressor, self).evaluate(dataset, **kwargs)
        return evaluated_metrics[metric_index]

    def _compute_target_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if not self.classifier:
            loss = tf.square(y_true - y_pred)
            return tf.reduce_mean(loss)

        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.reshape(y_true, shape=[-1])
        weights = tf.gather(self.class_weights, y_true)
        y_true = tf.one_hot(y_true, self.out_dim)
        y_pred = tf.reshape(y_pred, shape=[-1, self.out_dim])
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        loss = self.loss(y_true, y_pred)  # * weights
        return tf.reduce_mean(loss)

    def _compute_count_loss(
        self, counts: tf.Tensor, count_pred: tf.Tensor, count_mask
    ) -> tf.Tensor:
        count_mask = counts > 0
        count_mask = tf.reshape(count_mask, shape=[-1])
        relative_counts = tf.reshape(self._relative_abundance(counts), shape=[-1])[
            count_mask
        ]
        count_pred = tf.reshape(count_pred, shape=[-1])[count_mask]

        loss = tf.square(tf.math.log(relative_counts) - tf.math.log(count_pred))
        loss = tf.reduce_mean(loss)
        return loss

    def _compute_loss(
        self,
        model_inputs: tuple[tf.Tensor, tf.Tensor],
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: Union[
            tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        ],
        train_step: bool = True,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_counts, nuc_tokens, indicies, counts = model_inputs
        y_target, base_target = y_true

        (
            target_embeddings,
            count_pred,
            count_mask,
            y_pred,
            base_pred,
            nuc_mask,
            nuc_pred,
        ) = outputs
        target_loss = self._compute_target_loss(y_target, y_pred)

        counts = tf.cast(counts, dtype=tf.float32)
        counts = to_batch(counts, batch_counts)
        count_loss = self._compute_count_loss(counts, count_pred, count_mask)
        _, nuc_loss, encoder_loss = self.base_model._compute_loss(
            model_inputs, base_target, (base_target, base_pred, nuc_mask, nuc_pred)
        )

        return target_loss, count_loss, nuc_loss, encoder_loss

    def _compute_metric(
        self,
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: Union[
            tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        ],
    ):
        y_true, base_target = y_true

        (
            target_embeddings,
            count_pred,
            count_mask,
            y_pred,
            base_pred,
            nuc_mask,
            nuc_pred,
        ) = outputs
        if not self.classifier:
            y_true = y_true * self.scale + self.shift
            y_pred = y_pred * self.scale + self.shift
            self.metric_tracker.update_state(y_true, y_pred)
        else:
            y_true = tf.cast(y_true, dtype=tf.int32)
            y_pred = tf.keras.activations.softmax(y_pred)
            self.metric_tracker.update_state(y_true, y_pred)

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, (y_true, _) = data
        target_embeddings, count_pred, y_pred, base_pred, nuc_mask, nuc_pred = self(
            inputs, training=False
        )

        if not self.classifier:
            y_true = y_true * self.scale + self.shift
            y_pred = y_pred * self.scale + self.shift
        else:
            y_true = tf.cast(y_true, dtype=tf.int32)
            y_pred = tf.argmax(tf.keras.activations.softmax(y_pred), axis=-1)
        return y_pred, y_true

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            target_loss, count_mse, nuc_loss, encoder_loss = self._compute_loss(
                inputs, y, outputs
            )
            loss = target_loss + count_mse + nuc_loss + encoder_loss
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.target_tracker.update_state(target_loss)
        self.count_tracker.update_state(count_mse)

        self._compute_metric(y, outputs)
        self.base_model.encoder_tracker.update_state(encoder_loss)
        self.base_model.nuc_tracker.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "target_loss": self.target_tracker.result(),
            "count_mse": self.count_tracker.result(),
            "encoder_loss": self.base_model.encoder_tracker.result(),
            "nuc_loss": self.base_model.nuc_tracker.result(),
            self.metric_string: self.metric_tracker.result(),
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

        outputs = self(inputs, training=True)
        target_loss, count_mse, nuc_loss, encoder_loss = self._compute_loss(
            inputs, y, outputs
        )
        loss = target_loss + count_mse + nuc_loss + encoder_loss

        self.loss_tracker.update_state(loss)
        self.target_tracker.update_state(target_loss)
        self.count_tracker.update_state(count_mse)

        self._compute_metric(y, outputs)
        self.base_model.encoder_tracker.update_state(encoder_loss)
        self.base_model.nuc_tracker.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "target_loss": self.target_tracker.result(),
            "count_mse": self.count_tracker.result(),
            "encoder_loss": self.base_model.encoder_tracker.result(),
            "nuc_loss": self.base_model.nuc_tracker.result(),
            self.metric_string: self.metric_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def _relative_abundance(self, counts: tf.Tensor) -> tf.Tensor:
        count_sums = tf.reduce_sum(counts, axis=1, keepdims=True)
        rel_abundance = counts / count_sums
        return rel_abundance

    def mask_counts(self, counts, training=False):
        # select 15% of tokens to "mask" i.e. tokens to use to compute nuc_loss
        count_shape = tf.shape(counts)
        valid_mask = tf.cast(counts > 0, dtype=self.compute_dtype)
        random_mask = (
            create_random_mask(count_shape, percent=0.15, dtype=self.compute_dtype)
            * valid_mask
        )

        if False:
            # of the masked tokens, select 20% to either keep or change to
            # random token
            random_non_mask = (
                create_random_mask(count_shape, percent=0.2, dtype=self.compute_dtype)
                * random_mask
            )

            # of the 20% of masked tokens to either keep or change, select 50%  to keep
            # and 50% to change
            random_change = create_random_mask(
                count_shape, percent=0.5, dtype=self.compute_dtype
            )

            # tokens to keep the same
            random_keep = random_non_mask * random_change

            # tokens to randomly change
            random_change = (1 - random_keep) * valid_mask * random_non_mask

            # step 1: change all random_mask positions to <MASK> token
            masked_input = counts * (1 - random_mask)

            # step 2: change 10% of <MASK> tokens back to original token
            masked_input = (
                masked_input + counts * random_keep * random_mask * valid_mask
            )

            # step 3: change 10% of <MASK> tokens to random token
            random_tokens = tf.random.uniform(
                tf.shape(counts), minval=0, maxval=1, dtype=self.compute_dtype
            )

            # step 4: create masked input
            masked_input = (
                masked_input + random_tokens * random_change * random_mask * valid_mask
            )
            counts = masked_input

        # convert random_mask to boolean mask
        random_mask = random_mask > 0
        return counts, random_mask

    def _compute_count_embeddings(
        self,
        tensor: tf.Tensor,
        relative_abundances: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        count_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        count_embeddings = tensor + self.count_pos(tensor) * (1 + relative_abundances)
        count_embeddings = self.count_encoder(
            count_embeddings, mask=attention_mask, training=training
        )

        count_pred = self.count_out(count_embeddings)
        large_neg_num = -1e9

        mask = tf.cast(attention_mask, dtype=tf.float32)  # [B, T, 1]
        count_pred += (1.0 - mask) * large_neg_num  # Mask padding tokens
        count_pred = tf.squeeze(count_pred, axis=-1)
        count_pred = tf.nn.softmax(count_pred, axis=-1)

        # count_mask = tf.reshape(count_mask, shape=[-1])
        # count_pred = tf.reshape(count_pred, shape=[-1])[count_mask]

        return count_embeddings, count_pred

    def _compute_target_embeddings(
        self,
        tensor: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        # target_embeddings = self.target_encoder(
        #     tensor, mask=attention_mask, training=training
        # )
        target_embeddings = tensor
        target_out = self.attention_pooling(target_embeddings, mask=attention_mask)
        target_out = self.target_ff(target_out)
        return target_embeddings, target_out

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> Union[
        tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        # keras cast all input to float so we need to manually cast to expected type
        batch_counts, _, _, counts = inputs
        counts = tf.cast(counts, dtype=self.compute_dtype)

        counts = to_batch(counts, batch_counts)
        count_mask = float_mask(counts, dtype=self.compute_dtype)
        rel_abundance = self._relative_abundance(counts)

        count_attention_mask = count_mask
        base_embeddings, base_pred, nuc_mask, nuc_pred = self.base_model(
            inputs, training=(training and self.base_model.trainable)
        )

        # base_embeddings will always be float32
        base_embeddings = tf.cast(base_embeddings, dtype=self.compute_dtype)
        rel_abundance, count_mask = self.mask_counts(rel_abundance, training=training)
        count_gated_embeddings, count_pred = self._compute_count_embeddings(
            base_embeddings,
            rel_abundance,
            attention_mask=count_attention_mask,
            count_mask=count_mask,
            training=training,
        )

        # add percentage to base
        # count_embeddings = base_embeddings + self._rezero_a * count_gated_embeddings
        count_embeddings = count_gated_embeddings

        target_embeddings, target_out = self._compute_target_embeddings(
            count_embeddings, attention_mask=count_attention_mask, training=training
        )

        return (
            target_embeddings,
            count_pred,
            count_mask,
            target_out,
            base_pred,
            nuc_mask,
            nuc_pred,
        )

    def base_embeddings(
        self, inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> Union[
        tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        count_mask = float_mask(counts, dtype=tf.int32)
        rel_abundance = self._relative_abundance(counts)

        # account for <SAMPLE> token
        count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        rel_abundance = tf.pad(
            rel_abundance, [[0, 0], [1, 0], [0, 0]], constant_values=1
        )
        base_embeddings = self.base_model.base_embeddings((tokens, counts))

        return base_embeddings

    def base_gradient(
        self, inputs: tuple[tf.Tensor, tf.Tensor], base_embeddings
    ) -> Union[
        tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        count_mask = float_mask(counts, dtype=tf.int32)
        rel_abundance = self._relative_abundance(counts)

        # account for <SAMPLE> token
        count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        rel_abundance = tf.pad(
            rel_abundance, [[0, 0], [1, 0], [0, 0]], constant_values=1
        )
        count_attention_mask = count_mask

        count_gated_embeddings, count_pred = self._compute_count_embeddings(
            base_embeddings,
            rel_abundance,
            attention_mask=count_attention_mask,
        )
        # count_embeddings = base_embeddings + count_gated_embeddings * self._count_alpha
        count_embeddings = count_gated_embeddings

        target_embeddings, target_out = self._compute_target_embeddings(
            count_embeddings, attention_mask=count_attention_mask
        )

        return self.target_activation(target_out)

    def asv_embeddings(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> Union[
        tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        count_mask = float_mask(counts, dtype=tf.int32)
        rel_abundance = self._relative_abundance(counts)

        # account for <SAMPLE> token
        if self.add_token:
            count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
            rel_abundance = tf.pad(
                rel_abundance, [[0, 0], [1, 0], [0, 0]], constant_values=1
            )
        asv_embeddings = self.base_model.asv_embeddings(
            (tokens, counts), training=False
        )

        return asv_embeddings

    def asv_gradient(self, inputs, asv_embeddings):
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        count_mask = float_mask(counts, dtype=tf.int32)
        rel_abundance = self._relative_abundance(counts)

        # account for <SAMPLE> token
        if self.add_token:
            count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
            rel_abundance = tf.pad(
                rel_abundance, [[0, 0], [1, 0], [0, 0]], constant_values=1
            )
        count_attention_mask = count_mask
        base_embeddings = self.base_model.asv_gradient(
            (tokens, counts), asv_embeddings=asv_embeddings
        )

        count_gated_embeddings, count_pred = self._compute_count_embeddings(
            base_embeddings,
            rel_abundance,
            attention_mask=count_attention_mask,
            training=False,
        )
        # count_embeddings = base_embeddings + count_gated_embeddings
        count_embeddings = count_gated_embeddings

        target_embeddings, target_out = self._compute_target_embeddings(
            count_embeddings, attention_mask=count_attention_mask, training=False
        )

        return self.target_activation(target_out)

    def get_config(self):
        config = super(SequenceRegressor, self).get_config()
        config.update(
            {
                "token_limit": self.token_limit,
                "base_output_dim": self.base_output_dim,
                "shift": self.shift,
                "scale": self.scale,
                "dropout_rate": self.dropout_rate,
                "embedding_dim": self.embedding_dim,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "intermediate_activation": self.intermediate_activation,
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "freeze_base": self.freeze_base,
                "penalty": self.penalty,
                "nuc_penalty": self.nuc_penalty,
                "max_bp": self.max_bp,
                "is_16S": self.is_16S,
                "vocab_size": self.vocab_size,
                "out_dim": self.out_dim,
                "classifier": self.classifier,
                "add_token": self.add_token,
                "class_weights": self.class_weights,
                "asv_dropout_rate": self.asv_dropout_rate,
                "accumulation_steps": self.accumulation_steps,
                "scale_losses": self.scale_losses,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["base_model"] = tf.keras.saving.deserialize_keras_object(
            config["base_model"]
        )
        model = cls(**config)
        return model
