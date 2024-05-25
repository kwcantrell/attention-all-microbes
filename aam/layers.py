import tensorflow as tf
import tensorflow_models as tfm
from aam.sequence_utils import (
    add_random_sequences, add_random_seq_and_mask, random_sequences_mask, compute_pca_proj
)


@tf.keras.saving.register_keras_serializable(
    package="NucleotideEmbedding"
)
class NucleotideEmbedding(tf.keras.layers.Layer):
    def __init__(
            self,
            token_dim,
            max_bp,
            pca_hidden_dim,
            pca_heads,
            pca_layers,
            attention_heads,
            attention_layers,
            dff,
            dropout_rate,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.token_dim = token_dim
        self.max_bp = max_bp
        self.pca_hidden_dim = pca_hidden_dim
        self.pca_heads = pca_heads
        self.pca_layers = pca_layers
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.random_generator = (
            tf.random.Generator.from_non_deterministic_state()
        )

        self.emb_layer = tf.keras.layers.Embedding(
            8,
            512,
            input_length=max_bp,
            embeddings_initializer=tf.keras.initializers.GlorotNormal()
        )
        self.pca_layer = PCAProjector(
            hidden_dim=128,
            num_heads=8,
            num_layers=pca_layers,
            dropout=dropout_rate
        )
        self.ff_clr = tf.keras.layers.Dense(
            128, 'relu', use_bias=True
        )
        self.attention_layer = tfm.nlp.models.TransformerEncoder(
            num_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
            dropout_rate=dropout_rate,
            norm_first=True,
            activation='relu',
        )

    def build(self, input_shape):
        def add_seq_and_count_pad(seq, rclr):
            seq = tf.pad(
                seq,
                [
                    [0, 0],
                    [0, 1],
                    [0, 0]
                ],
                constant_values=7
            )
            rclr = tf.pad(
                rclr,
                paddings=[
                    [0, 0],
                    [0, 1]
                ],
                constant_values=0
            )
            return (seq, rclr)

        self.add_random_sequences = tf.function(add_random_sequences)
        self.add_random_seq_and_mask = tf.function(add_random_seq_and_mask)
        self.random_sequences_mask = tf.function(random_sequences_mask)
        self.add_seq_and_count_pad = add_seq_and_count_pad

    def call(
        self,
        inputs,
        mask=None,
        training=None,
        include_random=True,
        seq_mask_rate=0.
    ):
        seq, rclr = tf.nest.flatten(inputs, expand_composites=True)
        if training and include_random:
            seeds = self.random_generator.make_seeds(1)
            seq = self.random_sequences_mask(seq, seeds, seq_mask_rate)

        seq, rclr = tf.nest.flatten(self.add_seq_and_count_pad(seq, rclr))
        mask = tf.cast(
            tf.not_equal(
                seq,
                0
            ),
            dtype=tf.float32
        )
        attention_mask = tf.cast(
            tf.matmul(
                mask,
                mask,
                transpose_b=True
            ),
            dtype=tf.bool
        )
        output = self.emb_layer(seq)

        output = self.pca_layer(output)
        output = self.attention_layer(
            output,
            attention_mask=attention_mask,
            training=training
        )
        return output

    def get_config(self):
        base_config = super().get_config()
        config = {
            "token_dim": self.token_dim,
            "max_bp": self.max_bp,
            "pca_hidden_dim": self.pca_hidden_dim,
            "pca_heads": self.pca_heads,
            "pca_layers": self.pca_layers,
            "attention_heads": self.attention_heads,
            "attention_layers": self.attention_layers,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(
    package="PCAProjector"
)
class PCAProjector(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            num_layers,
            dropout,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.pca_layer = MultiHeadPCAProjection(
            hidden_dim,
            num_heads,
            dropout
        )
        self.ff = tf.keras.layers.Dense(hidden_dim)
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.point = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(
                max_length=input_shape[-2],
                seq_axis=2
        )

    def call(self, inputs):
        output = self.ff(inputs)
        output = output + self.pos_emb(output)
        pca_output = self.pca_layer(output)
        pca_output = self.dropout_layer(pca_output)
        pca_output = tf.squeeze(self.point(pca_output), axis=-1)
        return pca_output

    def get_config(self):
        base_config = super().get_config()

        config = {
            "hidden_dim":  self.hidden_dim,
            "num_heads":  self.num_heads,
            "num_layers":  self.num_layers,
            "dropout":  self.dropout,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(
    package="MultiHeadPCAProjection"
)
class MultiHeadPCAProjection(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            dropout,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm = tf.keras.layers.LayerNormalization(axis=-2)
        self.compute_pca_proj = tf.function(compute_pca_proj, jit_compile=True)
        self.head_size = self.hidden_dim // self.num_heads

    def call(self, inputs):
        output = inputs
        pca = self.compute_pca_proj(
            output,
            self.hidden_dim,
            self.num_heads,
            self.head_size
        )
        return pca

    def get_config(self):
        base_config = super().get_config()
        config = {
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(
    package="ReadHead"
)
class ReadHead(tf.keras.layers.Layer):
    def __init__(
            self,
            max_bp,
            hidden_dim,
            num_heads,
            num_layers,
            output_dim,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.max_bp = max_bp
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.reg_out = tf.keras.Sequential([
            tf.keras.layers.Dense(
                1024,
                activation='relu',
                use_bias=True
            ),
            tf.keras.layers.Dense(
                output_dim,
                use_bias=True
            ),
        ])

        self.attention_out = tf.keras.Sequential([
            tf.keras.layers.Dense(
                1024,
                activation='relu',
                use_bias=True
            ),
            tf.keras.layers.Dense(
                max_bp*6,
                activation='relu',
                use_bias=True
            ),
        ])
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(
            max_length=max_bp,
            seq_axis=2
        )
        self.nuc_out = tf.keras.layers.Dense(
            6,
            use_bias=True
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "max_bp": self.max_bp,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
        }
        return {**base_config, **config}

    def call(self, inputs):
        batch_dim = tf.shape(inputs)[0]
        num_seq = tf.shape(inputs)[1] - 1

        reg_out = inputs[:, -1, :]
        reg_out = self.reg_out(inputs[:, -1, :])
        
        seq_out = inputs[:, :-1, :]
        seq_out = tf.reshape(
            self.attention_out(seq_out),
            shape=[batch_dim, num_seq, self.max_bp, 6]
        )
        seq_out = self.nuc_out(seq_out + self.pos_emb(seq_out))

        return (reg_out, seq_out)
