from __future__ import annotations

from typing import Union

import tensorflow as tf
import tensorflow_models as tfm

from aam.losses import PairwiseLoss, TripletLoss
from aam.models import SequenceEncoder

# from aam.models.attention_pooling import AttentionPooling
from aam.models.base_sequence_encoder import BaseSequenceEncoder
from aam.models.multihead_attention_pooling import MultiHeadAttentionPooling
from aam.models.transformers import TransformerEncoder
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
        nucleotide_encoder=None,
        pairwise_loss_type="mse",
        normalize_outputs=True,
        unifrac_encoder=None,
        use_residual_connections=False,
        use_residual_pool=None,
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
        self.nucleotide_encoder = nucleotide_encoder
        self.pairwise_loss_type = pairwise_loss_type
        self.normalize_outputs = normalize_outputs
        self.use_residual_connections = use_residual_connections
        if use_residual_pool is None:
            use_residual_pool = use_residual_connections
        self.use_residual_pool = use_residual_pool

        self.unifrac_encoder = unifrac_encoder
        if self.unifrac_encoder is None:
            self.unifrac_encoder = SequenceEncoder(
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
                nucleotide_encoder=self.nucleotide_encoder,
                pairwise_loss_type=self.pairwise_loss_type,
                normalize_outputs=self.normalize_outputs,
                use_residual_connections=self.use_residual_connections,
                use_residual_pool=self.use_residual_pool,
            )

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pairwise_loss = PairwiseLoss(self.pairwise_loss_type)
        self.triplet_loss = TripletLoss()
        self.unifrac_tracker = tf.keras.metrics.Mean(name="unifrac_loss")
        self.denoise_tracker = tf.keras.metrics.Mean(name="denoised_loss")

        self.nuc_loss = tf.keras.losses.CategoricalCrossentropy(reduction="none")
        self.nuc_tracker = tf.keras.metrics.Mean(name="nuc_loss")

        self.gradient_accumulator = GradientAccumulator(self.accumulation_steps)
        self.loss_scaler = LossScaler(self.gradient_accumulator.accum_steps)

    def build(self, input_shape):
        self._rezero = self.add_weight(
            name="rezero_alpha",
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32,
        )
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(
            self.token_limit,
            seq_axis=1,
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
        )
        self.denoise_encoder = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
            normalize_outputs=self.normalize_outputs,
            use_residual_connections=self.use_residual_connections,
            name="encoder",
        )
        self.attention_pooling = MultiHeadAttentionPooling(
            self.normalize_outputs,
            num_heads=self.attention_heads,
            use_residual_connections=self.use_residual_pool,
        )

        self.denoiser_ff = tf.keras.layers.Dense(self.output_dim, dtype=tf.float32)
        super(UnifracDenoiser, self).build(input_shape)

    def _embeddings(self, tensor, mask=None, training=False):
        encoder_pred = self.attention_pooling(tensor, mask=mask, training=training)
        encoder_pred = self.denoiser_ff(encoder_pred)
        return encoder_pred

    def _compute_unifrac_loss(self, y_true: tf.Tensor, encoder_embeddings: tf.Tensor) -> tf.Tensor:
        return self._unifrac_loss((y_true, encoder_embeddings))

    def _unifrac_loss(self, inputs):
        y_true, encoder_embeddings = inputs
        loss = self.pairwise_loss(y_true, encoder_embeddings)
        loss = tf.reduce_mean(loss)
        return loss

    def _compute_loss(
        self,
        model_inputs: tuple[tf.Tensor, tf.Tensor],
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_counts, nuc_tokens, indicies, counts = model_inputs
        _, denoised_embeddings, unifrac_embeddings = outputs

        nuc_loss = tf.reduce_sum(self.unifrac_encoder.base_encoder.losses)

        shape = tf.shape(y_true)
        batch_dim = shape[0]
        group_dim = shape[-1]
        groups = batch_dim // group_dim
        y_true = tf.reshape(y_true, shape=[groups, group_dim, group_dim])
        unifrac_embeddings = tf.reshape(unifrac_embeddings, shape=[groups, group_dim, self.embedding_dim])
        unifrac_loss = tf.map_fn(
            self._unifrac_loss,
            (y_true, unifrac_embeddings),
            fn_output_signature=tf.float32,
        )
        unifrac_loss = tf.reduce_mean(unifrac_loss)

        denoised_embeddings = tf.unstack(tf.reshape(denoised_embeddings, shape=[groups, group_dim, self.embedding_dim]))
        denoise_loss = self.triplet_loss(denoised_embeddings[0], denoised_embeddings[1])

        denoise_loss = tf.reduce_mean(denoise_loss)

        loss = unifrac_loss + denoise_loss

        if self.train_nuc_encoder:
            print("add nuc loss")
            loss += nuc_loss
        return loss, nuc_loss, unifrac_loss, denoise_loss

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        embeddings, denoise_unifrac_embeddings, unifrac_embeddings = self.call(inputs, training=False)

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
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, nuc_loss, unifrac_loss, denoise_loss = self._compute_loss(inputs, encoder_target, outputs)

            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.unifrac_tracker.update_state(unifrac_loss)
        self.denoise_tracker.update_state(denoise_loss)
        self.nuc_tracker.update_state(nuc_loss)

        return {
            "loss": self.loss_tracker.result(),
            "unifrac_loss": self.unifrac_tracker.result(),
            "denoise_loss": self.denoise_tracker.result(),
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
        loss, nuc_loss, unifrac_loss, denoise_loss = self._compute_loss(inputs, encoder_target, outputs)
        self.loss_tracker.update_state(loss)
        self.unifrac_tracker.update_state(unifrac_loss)
        self.denoise_tracker.update_state(denoise_loss)
        self.nuc_tracker.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "unifrac_loss": self.unifrac_tracker.result(),
            "denoise_loss": self.denoise_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def call(
        self,
        inputs,
        include_bert_random_mask: bool = True,
        return_unifrac_pred: bool = True,
        return_counts: bool = False,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        sample_embeddings, counts, unifrac_pred = self.unifrac_encoder(
            inputs, include_bert_random_mask=include_bert_random_mask, return_counts=True, training=training
        )
        sample_embeddings = sample_embeddings + tf.cast(self._rezero, dtype=self.compute_dtype) * self.pos_emb(
            sample_embeddings
        )

        count_mask = tf.cast(counts > 0, dtype=self.compute_dtype)
        denoised_sample_embeddings = self.denoise_encoder(sample_embeddings, mask=count_mask, training=training)
        denoised_pred = self._embeddings(denoised_sample_embeddings, count_mask, training=training)
        print("UniFracDenoiser exit...")
        if return_unifrac_pred:
            if not return_counts:
                return denoised_sample_embeddings, denoised_pred, unifrac_pred
            else:
                return denoised_sample_embeddings, denoised_pred, unifrac_pred, counts
        else:
            if not return_counts:
                return denoised_sample_embeddings, denoised_pred
            else:
                return denoised_sample_embeddings, denoised_pred, counts

    def asv_embeddings(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens = inputs
        sample_embeddings = self.unifrac_encoder.base_encoder(tokens, training=False)
        return sample_embeddings

    @property
    def train_nuc_encoder(self):
        return self.unifrac_encoder.train_nuc_encoder

    @train_nuc_encoder.setter
    def train_nuc_encoder(self, flag: bool):
        self.unifrac_encoder.train_nuc_encoder = flag

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
                "nucleotide_encoder": self.nucleotide_encoder,
                "normalize_outputs": self.normalize_outputs,
                "use_residual_connections": self.use_residual_connections,
                "unifrac_encoder": tf.keras.saving.serialize_keras_object(self.unifrac_encoder),
                "use_residual_pool": self.use_residual_pool,
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        config["unifrac_encoder"] = tf.keras.saving.deserialize_keras_object(config["unifrac_encoder"])
        model = cls(**config)

        if input_shape is not None:
            model.build(input_shape)
        return model
