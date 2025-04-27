from __future__ import annotations

import tensorflow as tf

from aam.layers import ASVEncoder
from aam.losses import PairwiseLoss, _pairwise_cosine_distance, _pairwise_distances
from aam.models.feedforward import FeedForward


@tf.keras.saving.register_keras_serializable(package="NucleotideEncoderV6")
class NucleotideEncoderV6(tf.keras.Model):
    def __init__(
        self,
        embedding_dim: int,
        max_bp: int,
        dropout_rate: float,
        intermediate_activation: str = "gelu",
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 256,
        include_pos_emb: bool = False,
        use_cls_tkn=False,
        pairwise_type="mse",
        **kwargs,
    ):
        super(NucleotideEncoderV6, self).__init__(**kwargs)
        print("Constructing V6 model")
        self.embedding_dim = embedding_dim
        self.max_bp = max_bp
        self.dropout_rate = dropout_rate
        self.intermediate_activation = intermediate_activation
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size

        self.pairwise_type = pairwise_type
        if self.pairwise_type == "mse":
            self.dist_fn = _pairwise_distances
        else:
            self.dist_fn = _pairwise_cosine_distance
        self.asv_loss = tf.keras.losses.MeanSquaredError()
        self.loss_tracker = tf.keras.metrics.Mean()
        self.nuc_tracker = tf.keras.metrics.Mean()

        self.asv_tracker = tf.keras.metrics.Mean()
        self.include_pos_emb = include_pos_emb
        self.use_cls_tkn = use_cls_tkn
        self.asv_encoder = ASVEncoder(
            self.max_bp,
            self.attention_heads,
            self.attention_layers,
            self.dropout_rate,
            self.intermediate_size,
            intermediate_activation=self.intermediate_activation,
            embedding_dim=self.embedding_dim,
            include_pos_emb=self.include_pos_emb,
            use_cls_tkn=self.use_cls_tkn,
            name="asv_encoder",
        )

        def extract_asv_embedding(x):
            if self.use_cls_tkn:
                return x[:, 0]
            else:
                return tf.reduce_mean(x, axis=1)

        self.asv_ff = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(extract_asv_embedding),
                FeedForward(),
                tf.keras.layers.Dense(self.embedding_dim, dtype=tf.float32),
            ],
            name="asv_ff",
        )

    def build(self, input_shape):
        if self.built:
            return
        print("Building NucleotideEncoderV6...")
        self.asv_encoder.build(input_shape)
        input_shape = self.asv_encoder.compute_output_shape(input_shape)
        self.asv_ff.build(input_shape)
        super(NucleotideEncoderV6, self).build(input_shape)

    def compile(
        self,
        pairwise_type="mse",
        **kwargs,
    ):
        super().compile(**kwargs)
        self.pairwise_type = pairwise_type

    def build_graph(self, input_shape):
        """Builds graph

        Args:
            input_shape (tuple): A shape tuple (integers), not including the batch size.
        """
        super(NucleotideEncoderV6, self).build((None,) + input_shape)
        x = tf.keras.layers.Input(shape=(input_shape))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def predict_step(self, data):
        inputs, asv_ids = data
        asv_embeddings = self(inputs, training=False)
        if self.pairwise_type != "mse":
            print("normalizing embeddings!!!")
            asv_embeddings = tf.linalg.l2_normalize(asv_embeddings, axis=-1)
        return asv_embeddings, asv_ids

    def compute_distances(self, embeddings):
        distances = self.dist_fn(embeddings)
        mask = tf.linalg.band_part(tf.ones_like(distances, dtype=tf.bool), 0, 0)
        distances = distances * mask
        return distances + tf.transpose(distances)

    def _compute_loss(self, y_true, embeddings):
        distances = _pairwise_cosine_distance(embeddings)
        y_true = tf.expand_dims(y_true, axis=-1)
        distances = tf.expand_dims(distances, axis=-1)
        mask = y_true > 0
        distances._keras_mask = mask
        loss = self.asv_loss(y_true, distances)
        return loss

    def train_step(self, data):
        inputs, y_true = data
        with tf.GradientTape() as tape:
            output = self(inputs, return_randomize=True, training=True)
            asv_loss = self._compute_loss(y_true, output)
            nuc_loss = tf.reduce_sum(self.losses)
            unscaled_loss = asv_loss + nuc_loss
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(unscaled_loss)
            else:
                loss = unscaled_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(unscaled_loss)
        self.nuc_tracker.update_state(nuc_loss)
        self.asv_tracker.update_state(asv_loss)
        output_trackers = {
            "loss": self.loss_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "asv_loss": self.asv_tracker.result(),
            "lr": self.optimizer.lr,
        }
        return output_trackers

    def test_step(self, data):
        inputs, y_true = data

        output = self(inputs, return_randomize=True, training=False)
        asv_loss = self._compute_loss(y_true, output)
        nuc_loss = tf.reduce_sum(self.losses)
        loss = asv_loss + nuc_loss
        self.loss_tracker.update_state(loss)
        self.nuc_tracker.update_state(nuc_loss)
        self.asv_tracker.update_state(asv_loss)
        output_trackers = {
            "loss": self.loss_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "asv_loss": self.asv_tracker.result(),
            "lr": self.optimizer.learning_rate,
        }
        return output_trackers

    def call(self, inputs, return_randomize=False, training: bool = False) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        training = training and self.trainable

        embeddings = self.asv_encoder(inputs, training=training)
        asv_embeddings = self.asv_ff(embeddings)
        return asv_embeddings

    def get_config(self):
        config = super(NucleotideEncoderV6, self).get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "max_bp": self.max_bp,
                "dropout_rate": self.dropout_rate,
                "intermediate_activation": self.intermediate_activation,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "include_pos_emb": self.include_pos_emb,
                "use_cls_tkn": self.use_cls_tkn,
                "pairwise_type": self.pairwise_type,
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

        model = cls(**config)

        if input_shape is not None:
            model.build_graph(tuple(input_shape[1:]))
        return model
