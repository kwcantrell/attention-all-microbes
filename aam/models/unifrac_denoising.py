from __future__ import annotations

from typing import Union

import tensorflow as tf

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
        denoised_embeddings, unifrac_embeddings = outputs

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

        tokens, batch_indicies, asv_indicies, counts = inputs

        batch_indicies = tf.cast(batch_indicies, dtype=tf.int32)
        asv_indicies = tf.cast(asv_indicies, dtype=tf.int32)

        asv_embeddings = self.asv_encoder(tokens, training=False)
        asv_embeddings = tf.gather(asv_embeddings, asv_indicies)

        shape = tf.shape(encoder_target)
        batch_dim = shape[0]
        group_dim = shape[-1]

        # group_input = [self._group_embeddings(asv_embeddings, (batch_indicies, counts), i, group_dim) for i in range(2)]
        # group_input = tf.map_fn(
        #     lambda i: self._group_embeddings(asv_embeddings, (batch_indicies, counts), i, group_dim),
        #     tf.range(2),
        #     fn_output_signature=(tf.float32, tf.int32),
        #     # fn_output_signature=(
        #     #     tf.RaggedTensorSpec(shape=[None, None, self.embedding_dim], dtype=self.compute_dtype),
        #     #     tf.RaggedTensorSpec(shape=[None, None, 1], dtype=tf.int32),
        #     # ),
        # )
        def run_group(i):
            groupt_input = self._group_embeddings(asv_embeddings, (batch_indicies, counts), i, group_dim)

            return self.call(groupt_input, training=True)

        with tf.GradientTape() as tape:
            # # denoise_embeddings, unifrac_embeddings = [], []
            # # for i in range(2):
            # #     _den, _uni = self.call(group_input[i], training=True)
            # #     denoise_embeddings.append(_den)
            # #     unifrac_embeddings.append(_uni)
            # outputs = tf.map_fn(
            #     lambda i: self.call(group_input[i], training=True),
            #     tf.range(2),
            #     fn_output_signature=(tf.float32, tf.float32),
            # )
            # denoise_embeddings, unifrac_embeddings = tf.nest.flatten(outputs)

            # denoise_embeddings = tf.concat(denoise_embeddings, axis=0)
            # unifrac_embeddings = tf.concat(unifrac_embeddings, axis=0)
            outputs = tf.map_fn(
                run_group,
                tf.range(2),
                fn_output_signature=(
                    tf.TensorSpec(shape=[None, self.embedding_dim], dtype=tf.float32),
                    tf.TensorSpec(shape=[None, self.embedding_dim], dtype=tf.float32),
                ),
            )
            denoise_embeddings, unifrac_embeddings = tf.nest.flatten(outputs)
            denoise_embeddings = tf.reshape(denoise_embeddings, shape=(-1, self.embedding_dim))
            unifrac_embeddings = tf.reshape(unifrac_embeddings, shape=(-1, self.embedding_dim))

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

    def _batch_embeddings(self, asv_embeddings, batch_indicies, counts):
        batch_shape = tf.reduce_max(batch_indicies[:, 0]) + 1
        max_unique = tf.reduce_max(batch_indicies[:, 1]) + 1
        batch_embeddings = tf.scatter_nd(batch_indicies, asv_embeddings, shape=[batch_shape, max_unique, self.embedding_dim])
        counts = tf.scatter_nd(batch_indicies, counts, shape=[batch_shape, max_unique, 1])
        return batch_embeddings, counts

    def _extract_asv_embeddings(self, inputs):
        tokens, batch_indicies, asv_indicies, counts = inputs

        batch_indicies = tf.cast(batch_indicies, dtype=tf.int32)
        asv_indicies = tf.cast(asv_indicies, dtype=tf.int32)

        asv_embeddings = self.asv_encoder(tokens, training=False)
        asv_embeddings = tf.gather(asv_embeddings, asv_indicies)

        return self._batch_embeddings(asv_embeddings, batch_indicies, counts)

    def _group_embeddings(self, asv_embeddings, inputs, group, samples_per_group):
        batch_indicies, counts = inputs

        batch_indicies = tf.cast(batch_indicies, dtype=tf.int32)
        row_indices = tf.expand_dims(tf.range(group, group + samples_per_group, 1, dtype=tf.int32), axis=0)
        row_mask = tf.math.reduce_any(row_indices == batch_indicies[:, :1], axis=-1)
        row_indices = tf.squeeze(tf.where(row_mask))

        asv_embeddings = tf.gather(asv_embeddings, row_indices)
        batch_indicies = tf.gather(batch_indicies, row_indices)
        counts = tf.gather(counts, row_indices)

        group_size = tf.shape(batch_indicies)[0]
        group_shift = tf.reduce_min(batch_indicies[:, 0])
        group_index_shift = tf.repeat([[group_shift, 0]], repeats=group_size, axis=0)
        return self._batch_embeddings(asv_embeddings, batch_indicies - group_index_shift, counts)

    def call(
        self,
        inputs,
        attention_mask=None,
        return_asv_embeddings: bool = False,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable

        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 2:
                asv_embeddings, counts = self._extract_asv_embeddings(inputs)
            else:
                asv_embeddings, counts = inputs
            attention_mask = tf.cast(counts > 0, dtype=self.compute_dtype)
        else:
            asv_embeddings = inputs

        # ensure asv_embeddings match the expected compute type
        asv_embeddings = tf.cast(asv_embeddings, dtype=self.compute_dtype)
        asv_embeddings, unifrac_embeddings = self.unifrac_encoder(
            asv_embeddings, attention_mask=attention_mask, training=training
        )
        asv_embeddings, denoised_unifrac_embeddings = self.unifrac_denoiser(
            asv_embeddings, attention_mask=attention_mask, training=training
        )
        print("UniFracDenoiser exit...", self.trainable)
        if return_asv_embeddings:
            return asv_embeddings, counts
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
        token_shape = tf.TensorShape([None, 150])
        batch_indicies = tf.TensorShape([None, 2])
        indicies_shape = tf.TensorShape([None])
        count_shape = tf.TensorShape([None, 1])
        if input_shape is not None:
            model.build([token_shape, batch_indicies, indicies_shape, count_shape])
        return model
