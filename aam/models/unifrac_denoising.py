from __future__ import annotations

from typing import Union

import tensorflow as tf
import tensorflow_models as tfm

from aam.losses import PairwiseLoss, triplet_loss
from aam.models.multihead_attention_pooling import MultiHeadAttentionPooling
from aam.models.transformers import TransformerEncoder
from aam.models.unifrac_encoder import UnifracEncoder
from aam.models.utils import sort_using_counts, to_batch
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler
from aam.utils import float_mask


@tf.keras.saving.register_keras_serializable(package="UnifracDenoiser")
class UnifracDenoiser(tf.keras.Model):
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
        asv_encoder=None,
        use_linear_bias=False,
        **kwargs,
    ):
        super(UnifracDenoiser, self).__init__(**kwargs)
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
        self.use_linear_bias = use_linear_bias

        if asv_encoder is None:
            raise Exception("UnifracDeniser is missing ASVEncoder")
        self.asv_encoder = asv_encoder

        if use_residual_pool is None:
            use_residual_pool = use_residual_connections
        self.use_residual_pool = use_residual_pool

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pairwise_loss = PairwiseLoss(self.pairwise_loss_type)
        self.triplet_loss = triplet_loss
        self.unifrac_tracker = tf.keras.metrics.Mean(name="unifrac_loss")
        self.denoise_tracker = tf.keras.metrics.Mean(name="denoised_loss")

        self.gradient_accumulator = GradientAccumulator(self.accumulation_steps)
        self.loss_scaler = LossScaler(self.gradient_accumulator.accum_steps)

    def build(self, input_shape):
        if self.built:
            print("UnifracDenoiser is already built")
            return

        self.unifrac_encoder = UnifracEncoder(
            output_dim=self.output_dim,
            token_limit=self.token_limit,
            encoder_type=self.encoder_type,
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
            pairwise_loss_type=self.pairwise_loss_type,
            normalize_outputs=self.normalize_outputs,
            use_residual_connections=self.use_residual_connections,
            use_residual_pool=self.use_residual_pool,
            use_linear_bias=self.use_linear_bias,
            name="unifrac_encoder",
        )

        self.unifrac_denoiser = UnifracEncoder(
            output_dim=self.output_dim,
            token_limit=self.token_limit,
            encoder_type=self.encoder_type,
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
            pairwise_loss_type=self.pairwise_loss_type,
            normalize_outputs=self.normalize_outputs,
            use_residual_connections=self.use_residual_connections,
            use_residual_pool=self.use_residual_pool,
            use_linear_bias=self.use_linear_bias,
            name="unifrac_denoiser",
        )

        super(UnifracDenoiser, self).build(input_shape)

    def _compute_unifrac_loss(self, unifrac_distances, unifrac_embeddings):
        shape = tf.shape(unifrac_distances)
        batch_dim = shape[0]
        group_dim = shape[-1]
        groups = batch_dim // group_dim
        unifrac_distances = tf.reshape(unifrac_distances, shape=[groups, group_dim, group_dim])
        unifrac_embeddings = tf.reshape(unifrac_embeddings, shape=[groups, group_dim, self.embedding_dim])
        unifrac_loss = tf.map_fn(
            lambda inputs: self.pairwise_loss(inputs[0], inputs[1]),
            (unifrac_distances, unifrac_embeddings),
            fn_output_signature=tf.float32,
        )
        unifrac_loss = tf.reduce_mean(unifrac_loss)
        return unifrac_loss

    def _compute_denoise_loss(self, denoised_embeddings):
        denoise_loss = self.triplet_loss(denoised_embeddings)
        return tf.reduce_mean(denoise_loss)

    def _compute_loss(
        self,
        unifrac_distances: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        denoised_embeddings, unifrac_embeddings = outputs

        unifrac_loss = self._compute_unifrac_loss(unifrac_distances, unifrac_embeddings)

        denoise_loss = self.triplet_loss(denoised_embeddings)
        denoise_loss = 0.1 * tf.reduce_mean(denoise_loss)

        loss = unifrac_loss + denoise_loss

        return loss, unifrac_loss, denoise_loss

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        denoise_unifrac_embeddings, unifrac_embeddings = self.call(inputs, training=False)

        return unifrac_embeddings, y

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

        group_dim = tf.shape(encoder_target)[-1]
        batch_counts, tokens, indicies, counts = inputs

        batch_counts = tf.reshape(batch_counts, shape=[-1, group_dim])

        asv_embeddings = self.asv_encoder(tokens, training=False)
        b1_batch_counts = batch_counts[0]

        b1_total = tf.reduce_sum(b1_batch_counts)
        b1_indices = indicies[:b1_total]
        b1_counts = counts[:b1_total]
        b1_embeddings, b1_mask = self._batch_embeddings(asv_embeddings, b1_batch_counts, b1_indices, b1_counts)

        b2_batch_counts = batch_counts[1]
        b2_indices = indicies[b1_total:]
        b2_counts = counts[b1_total:]
        b2_embeddings, b2_mask = self._batch_embeddings(asv_embeddings, b2_batch_counts, b2_indices, b2_counts)

        with tf.GradientTape() as tape:
            b1_denoise, b1_unifrac = self.call(b1_embeddings, attention_mask=b1_mask, training=True)
            b2_denoise, b2_unifrac = self.call(b2_embeddings, attention_mask=b2_mask, training=True)

            denoise_embeddings = tf.concat([b1_denoise, b2_denoise], axis=0)
            unifrac_embeddings = tf.concat([b1_unifrac, b2_unifrac], axis=0)

            loss, unifrac_loss, denoise_loss = self._compute_loss(encoder_target, (denoise_embeddings, unifrac_embeddings))
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(unifrac_loss + denoise_loss)
        self.unifrac_tracker.update_state(unifrac_loss)
        self.denoise_tracker.update_state(denoise_loss)

        return {
            "loss": self.loss_tracker.result(),
            "unifrac_loss": self.unifrac_tracker.result(),
            "denoise_loss": self.denoise_tracker.result(),
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
        loss, unifrac_loss, denoise_loss = self._compute_loss(encoder_target, outputs)
        self.loss_tracker.update_state(loss)
        self.unifrac_tracker.update_state(unifrac_loss)
        self.denoise_tracker.update_state(denoise_loss)
        return {
            "loss": self.loss_tracker.result(),
            "unifrac_loss": self.unifrac_tracker.result(),
            "denoise_loss": self.denoise_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def _batch_embeddings(self, embeddings, batch_counts, indicies, counts):
        embeddings = tf.gather(embeddings, tf.cast(indicies, dtype=tf.int32))
        embeddings = to_batch(embeddings, batch_counts)

        counts = to_batch(counts, batch_counts)
        embeddings, counts = sort_using_counts(embeddings, counts)
        return embeddings, tf.cast(counts > 0, dtype=self.compute_dtype)

    def _extract_asv_embeddings(self, inputs):
        batch_counts, tokens, indicies, counts = inputs
        asv_embeddings = self.asv_encoder(tokens, training=False)
        return self._batch_embeddings(asv_embeddings, batch_counts, indicies, counts)

    def call(
        self, inputs, attention_mask=None, return_asv_embeddings: bool = False, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable

        if isinstance(inputs, (tuple, list)):
            asv_embeddings, attention_mask = self._extract_asv_embeddings(inputs)
        else:
            asv_embeddings = inputs

        asv_embeddings, unifrac_embeddings = self.unifrac_encoder(
            asv_embeddings, attention_mask=attention_mask, training=training
        )
        asv_embeddings, denoised_unifrac_embeddings = self.unifrac_denoiser(
            asv_embeddings, attention_mask=attention_mask, training=training
        )
        print("UniFracDenoiser exit...", self.trainable)
        if return_asv_embeddings:
            return asv_embeddings, attention_mask, denoised_unifrac_embeddings, unifrac_embeddings
        else:
            return denoised_unifrac_embeddings, unifrac_embeddings

    def asv_embeddings(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens = inputs
        sample_embeddings = self.unifrac_encoder.base_encoder(tokens, training=False)
        return sample_embeddings

    def get_config(self):
        config = super(UnifracDenoiser, self).get_config()
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
                "build_input_shape": self.get_build_config(),
                "asv_encoder": tf.keras.saving.serialize_keras_object(self.asv_encoder),
                "use_linear_bias": self.use_linear_bias,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        print("Reconstructing ASVEncoder...")
        asv_encoder = tf.keras.saving.deserialize_keras_object(config["asv_encoder"])
        asv_encoder.trainable = False
        config["asv_encoder"] = asv_encoder

        print("Constructing UnifracDenoser from config")
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)
        batch_counts = tf.TensorShape([None])
        token_shape = tf.TensorShape([None, 150])
        indicies_shape = tf.TensorShape([None])
        count_shape = tf.TensorShape([None, 1])
        if input_shape is not None:
            model.build([batch_counts, token_shape, indicies_shape, count_shape])
        return model
