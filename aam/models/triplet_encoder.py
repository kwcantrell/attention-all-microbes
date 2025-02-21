from __future__ import annotations

from typing import Union

import tensorflow as tf

# from aam.data_handlers.generator_dataset import batch_embeddings
from aam.losses import PairwiseLoss, categorical_triplet_loss
from aam.models.unifrac_encoder import UnifracEncoder
from aam.models.utils import sort_using_counts
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler


@tf.keras.saving.register_keras_serializable(package="TripletEncoder")
class TripletEncoder(tf.keras.Model):
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
        super(TripletEncoder, self).__init__(**kwargs)
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
        self.triplet_loss = categorical_triplet_loss
        self.unifrac_tracker = tf.keras.metrics.Mean(name="unifrac_loss")
        self.denoise_tracker = tf.keras.metrics.Mean(name="denoised_loss")

        self.gradient_accumulator = GradientAccumulator(self.accumulation_steps)
        self.loss_scaler = LossScaler(self.gradient_accumulator.accum_steps)

    def build(self, input_shape):
        if self.built:
            print("TripletEncoder is already built")
            return

        self.encoder = UnifracEncoder(
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

        super(TripletEncoder, self).build(input_shape)

    def _compute_loss(
        self,
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_asv_embeddings, denoised_embeddings = outputs

        denoise_loss = tf.reduce_mean(self.triplet_loss(denoised_embeddings))

        loss = denoise_loss
        return loss, denoise_loss

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        unifrac_embeddings, denoise_unifrac_embeddings = self.call(
            inputs, training=False
        )
        return denoise_unifrac_embeddings, y

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
            loss, denoise_loss = self._compute_loss(outputs)
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(denoise_loss)
        self.denoise_tracker.update_state(denoise_loss)

        return {
            "loss": self.loss_tracker.result(),
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

    def batch_embeddings(
        self, asv_embeddings, batch_indicies, counts, asv_indices=None
    ):
        emb_dim = tf.shape(asv_embeddings)[-1]
        if asv_indices is not None:
            asv_embeddings = tf.gather(asv_embeddings, asv_indices)
        batch_shape = tf.reduce_max(batch_indicies[:, 0]) + 1
        max_unique = tf.reduce_max(batch_indicies[:, 1]) + 1
        batch_embeddings = tf.scatter_nd(
            batch_indicies, asv_embeddings, shape=[batch_shape, max_unique, emb_dim]
        )
        counts = tf.scatter_nd(
            batch_indicies, counts, shape=[batch_shape, max_unique, 1]
        )
        return batch_embeddings, counts

    def extract_asv_embeddings(self, inputs, sort_counts=False):
        tokens, batch_indices, asv_indices, counts = inputs
        batch_indices = tf.cast(batch_indices, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)

        asv_embeddings = tf.cast(
            self.asv_encoder(tokens, training=False), dtype=self.compute_dtype
        )

        asv_embeddings, counts = self.batch_embeddings(
            asv_embeddings, batch_indices, counts, asv_indices
        )
        if not sort_counts:
            return asv_embeddings, counts

        asv_embeddings, counts = sort_using_counts(asv_embeddings, counts)
        return asv_embeddings, counts

    def call(
        self,
        inputs,
        return_asv_embeddings: bool = False,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable

        asv_embeddings, counts = self.extract_asv_embeddings(inputs)

        attention_mask = tf.cast(counts > 0, dtype=self.compute_dtype)

        batch_asv_embeddings, denoised_embeddings = self.encoder(
            asv_embeddings, attention_mask=attention_mask, training=training
        )

        print("TripletEncoder exit...", self.trainable)
        return batch_asv_embeddings, denoised_embeddings

    def get_config(self):
        config = super(TripletEncoder, self).get_config()
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
        token_shape = tf.TensorShape([None, 150])
        batch_indices = tf.TensorShape([None, 2])
        indicies_shape = tf.TensorShape([None])
        count_shape = tf.TensorShape([None, 1])
        if input_shape is not None:
            model.build([token_shape, batch_indices, indicies_shape, count_shape])
        return model
