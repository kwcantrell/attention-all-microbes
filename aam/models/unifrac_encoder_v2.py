from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.losses import PairwiseLoss
from aam.models.feedforward import FeedForward
from aam.models.transformers import TransformerEncoder


@tf.keras.saving.register_keras_serializable(package="UnifracEncoderV2")
class UnifracEncoderV2(tf.keras.Model):
    def __init__(self, **kwargs):
        super(UnifracEncoderV2, self).__init__(**kwargs)
        self.unifrac_loss = PairwiseLoss(use_mean_pairs=False)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.encoder = TransformerEncoder(
            num_layers=8,
            num_attention_heads=4,
            intermediate_size=1024,
            use_linear_bias=True,
            name="unifrac_encoder",
        )

    def build_graph(self, input_shape):
        """Builds graph

        Args:
            input_shape (tuple): A shape tuple (integers), not including the batch size.
        """
        if input_shape is None:
            input_shape = self._build_input_shape
        x = tf.keras.layers.Input(shape=(input_shape))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def build(self, input_shape):
        embedding_dim = input_shape[-1]
        self.asv_ff = tf.keras.Sequential(
            [
                FeedForward(),
                tf.keras.layers.Dense(embedding_dim, dtype=tf.float32),
            ],
            name="unifrac_ff",
        )
        super(UnifracEncoderV2, self).build(input_shape)
        self.built = True

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        return self(inputs, training=False), y

    def _compute_loss(self, y, output_embeddings):
        loss = self.unifrac_loss(y, output_embeddings)
        return tf.reduce_mean(loss)

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data

        with tf.GradientTape() as tape:
            output_embeddings = self(inputs, training=True)
            unscaled_loss = self._compute_loss(y, output_embeddings)
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(unscaled_loss)
            else:
                loss = unscaled_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(unscaled_loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        output_embeddings = self(inputs, training=False)
        loss = self._compute_loss(y, output_embeddings)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _get_dense_embeddings(self, inputs):
        sparse_indices, embeddings = inputs
        sparse_indices = tf.cast(sparse_indices, dtype=tf.int32)
        batch_dim = tf.reduce_max(sparse_indices[:, 0]) + 1
        seq_dim = tf.reduce_max(sparse_indices[:, 1]) + 1
        dense_embeddings = tf.scatter_nd(
            sparse_indices,
            embeddings,
            [batch_dim, seq_dim, tf.shape(embeddings)[-1]],
        )
        mask = tf.scatter_nd(
            sparse_indices,
            tf.ones([tf.shape(sparse_indices)[0], 1], dtype=self.compute_dtype),
            [batch_dim, seq_dim, 1],
        )
        return dense_embeddings, mask

    def call(
        self, inputs, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        dense_embeddings, mask = self._get_dense_embeddings(inputs)
        dense_embeddings = tf.cast(dense_embeddings, dtype=self.compute_dtype)

        encoder_output = self.encoder(
            dense_embeddings, mask=mask, training=training
        )
        ff_input = tf.reduce_sum(encoder_output, axis=1) / tf.reduce_sum(
            mask, axis=1
        )

        output = ff_input + tf.cast(
            self._rezero, dtype=self.compute_dtype
        ) * self.ff(ff_input)

        print("UnifracEncoderV2 exit...")
        return self.output_activation(output)

    def get_config(self):
        config = super(UnifracEncoderV2, self).get_config()
        config.update(
            {
                "num_encoder_layers": self.num_encoder_layers,
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "conv_blocks_per_layer": self.conv_blocks_per_layer,
                "build_input_shape": self.get_build_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        print("Constructing UnifracEncoderV2 from config")
        input_shape = None
        if "build_input_shape" in config:
            build_input_shape = config.pop("build_input_shape")
            input_shape = build_input_shape["input_shape"]

        model = cls(**config)
        model.build(input_shape)
        return model
