from __future__ import annotations

from typing import Union

import tensorflow as tf

# from aam.data_handlers.generator_dataset import batch_embeddings
from aam.losses import PairwiseLoss, triplet_loss
from aam.models.unifrac_encoder import UnifracEncoder
from aam.models.utils import sort_using_counts, to_batch
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler


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

        self.asv_input_ff = tf.keras.layers.Dense(self.embedding_dim, use_bias=True)

        super(UnifracDenoiser, self).build(input_shape)

    def _compute_unifrac_loss(self, unifrac_distances, unifrac_embeddings):
        shape = tf.shape(unifrac_distances)
        batch_dim = shape[0]
        group_dim = shape[-1]
        groups = batch_dim // group_dim
        unifrac_distances = tf.reshape(
            unifrac_distances, shape=[groups, group_dim, group_dim]
        )
        unifrac_embeddings = tf.reshape(
            unifrac_embeddings, shape=[groups, group_dim, self.embedding_dim]
        )

        def _unifrac_loss(inputs):
            uni_dist, uni_emb = inputs
            return tf.reduce_mean(self.pairwise_loss(uni_dist, uni_emb))

        losses = tf.map_fn(
            _unifrac_loss,
            [unifrac_distances, unifrac_embeddings],
            fn_output_signature=tf.float32,
        )

        return tf.reduce_mean(losses)

    def _compute_denoise_loss(self, denoised_embeddings):
        denoise_loss = self.triplet_loss(denoised_embeddings)
        return tf.reduce_mean(0.1 * denoise_loss)

    def _compute_loss(
        self,
        unifrac_distances: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        unifrac_embeddings, denoised_embeddings = outputs

        unifrac_loss = self._compute_unifrac_loss(unifrac_distances, unifrac_embeddings)
        denoise_loss = self._compute_denoise_loss(denoised_embeddings)

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
        unifrac_embeddings, denoise_unifrac_embeddings = self.call(
            inputs, training=False
        )

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

        shape = tf.shape(encoder_target)
        group_dim = shape[-1]

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, unifrac_loss, denoise_loss = self._compute_loss(
                encoder_target, outputs
            )
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

    def _group_embeddings(self, asv_embeddings, inputs, group, samples_per_group):
        batch_indices, asv_indices, counts = inputs

        batch_indices = tf.cast(batch_indices, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)
        group_rows = tf.expand_dims(
            tf.range(
                group * samples_per_group,
                group * samples_per_group + samples_per_group,
                1,
                dtype=tf.int32,
            ),
            axis=0,
        )
        row_mask = tf.math.reduce_any(
            batch_indices[:, :1] == group_rows, axis=-1, keepdims=True
        )
        group_shift = (
            tf.pad(
                tf.cast(row_mask, dtype=tf.int32),
                paddings=[[0, 0], [0, 1]],
                constant_values=0,
            )
            * group
            * samples_per_group
        )
        group_mask = tf.squeeze(row_mask, axis=-1)

        batch_indices = batch_indices[group_mask] - group_shift[group_mask]
        asv_indices = asv_indices[group_mask]
        counts = counts[group_mask]

        return self.batch_embeddings(asv_embeddings, batch_indices, counts, asv_indices)

    def _create_unifrac_embeddings(self, asv_embeddings, counts, training):
        attention_mask = tf.cast(counts > 0, dtype=self.compute_dtype)

        asv_embeddings, unifrac_embeddings = self.unifrac_encoder(
            asv_embeddings, attention_mask=attention_mask, training=training
        )

        asv_embeddings, counts = sort_using_counts(asv_embeddings, counts)
        attention_mask = tf.cast(counts > 0, dtype=self.compute_dtype)

        asv_embeddings, denoised_unifrac_embeddings = self.unifrac_denoiser(
            asv_embeddings, attention_mask=attention_mask, training=training
        )
        return asv_embeddings, unifrac_embeddings, denoised_unifrac_embeddings

    def call(
        self,
        inputs,
        return_asv_embeddings: bool = False,
        samples_per_group: int = None,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable

        tokens, batch_indices, asv_indices, counts = inputs

        tokens, batch_indices, asv_indices, counts = inputs
        asv_embeddings = tf.cast(
            self.asv_encoder(tokens, training=False), dtype=self.compute_dtype
        )
        asv_embeddings = self.asv_input_ff(asv_embeddings)

        batch_indices = tf.cast(batch_indices, dtype=tf.int32)
        asv_indices = tf.cast(asv_indices, dtype=tf.int32)

        if samples_per_group is not None:
            max_seq = tf.reduce_max(batch_indices[:, 1]) + 1

            def run_group(i):
                group_asv_embeddings, group_counts = self._group_embeddings(
                    asv_embeddings,
                    (batch_indices, asv_indices, counts),
                    i,
                    samples_per_group,
                )
                (
                    group_asv_embeddings,
                    unifrac_embeddings,
                    denoised_unifrac_embeddings,
                ) = self._create_unifrac_embeddings(
                    group_asv_embeddings, group_counts, training
                )

                group_seq = tf.shape(group_asv_embeddings)[1]
                return (
                    tf.pad(
                        group_asv_embeddings, [[0, 0], [0, max_seq - group_seq], [0, 0]]
                    ),
                    unifrac_embeddings,
                    denoised_unifrac_embeddings,
                )

            outputs = tf.map_fn(
                run_group,
                tf.range(2),
                fn_output_signature=(
                    tf.TensorSpec(
                        shape=[None, None, self.embedding_dim], dtype=self.compute_dtype
                    ),
                    tf.TensorSpec(shape=[None, self.embedding_dim], dtype=tf.float32),
                    tf.TensorSpec(shape=[None, self.embedding_dim], dtype=tf.float32),
                ),
            )
            batch_asv_embeddings, unifrac_embeddings, denoised_unifrac_embeddings = (
                tf.nest.flatten(outputs)
            )

            batch_shape = tf.reduce_max(batch_indices[:, 0]) + 1
            batch_asv_embeddings = tf.reshape(
                denoised_unifrac_embeddings, shape=(batch_shape, -1, self.embedding_dim)
            )
            denoised_unifrac_embeddings = tf.reshape(
                denoised_unifrac_embeddings, shape=(batch_shape, self.embedding_dim)
            )
            unifrac_embeddings = tf.reshape(
                unifrac_embeddings, shape=(batch_shape, self.embedding_dim)
            )
        else:
            batch_asv_embeddings, counts = self.batch_embeddings(
                asv_embeddings, batch_indices, counts, asv_indices
            )

            batch_asv_embeddings, unifrac_embeddings, denoised_unifrac_embeddings = (
                self._create_unifrac_embeddings(batch_asv_embeddings, counts, training)
            )

        print("UniFracDenoiser exit...", self.trainable)
        if return_asv_embeddings:
            return batch_asv_embeddings, counts
        else:
            return unifrac_embeddings, denoised_unifrac_embeddings

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
        token_shape = tf.TensorShape([None, 150])
        batch_indices = tf.TensorShape([None, 2])
        indicies_shape = tf.TensorShape([None])
        count_shape = tf.TensorShape([None, 1])
        if input_shape is not None:
            model.build([token_shape, batch_indices, indicies_shape, count_shape])
        return model
