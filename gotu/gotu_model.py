"""gotu_model.py"""

import biom
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm

from aam.callbacks import SaveModel
from aam.layers import InputLayer
from aam.nuc_model import BaseNucleotideModel
from aam.utils import LRDecrease


class TransferLearnNucleotideModel(BaseNucleotideModel):
    def __init__(
        self,
        base_model,
        dropout_rate,
        batch_size,
        max_bp,
        num_gotus,
        pca_hidden_dim,
        pca_heads,
        pca_layers,
        count_ff_dim=32,
        num_layers=2,
        num_attention_heads=8,
        dff=32,
        use_attention_loss=True,
        d_model=128,
        **kwargs,
    ):
        use_attention_loss = False
        super().__init__(
            batch_size=batch_size,
            use_attention_loss=use_attention_loss,
            **kwargs,
        )

        self.max_bp = max_bp
        self.dropout_rate = dropout_rate
        self.count_ff_dim = count_ff_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.dff = dff
        self.regresssion_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric_traker = tf.keras.metrics.Mean(name="loss")

        self.input_layer = InputLayer(name="input_layer")

        self.base_model = base_model
        self.base_model.trainable = False  # Freeze the base model

        self.decoder_embedding = tf.keras.layers.Embedding(
            input_dim=num_gotus + 4, output_dim=128, embeddings_initializer="uniform", input_length=1, name="decoder_embedding"
        )

        self.transformer_decoder = tfm.nlp.models.TransformerDecoder(
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=dff,
            norm_first=True,
            activation="relu",
        )

        self.dense_output = tf.keras.layers.Dense(num_gotus + 4)

    def call(self, inputs, training=False):
        encoder_inputs, decoder_inputs = self.input_layer(inputs)
        encoder_output = self.base_model(encoder_inputs, training=training)

        decoder_embeddings = self.decoder_embedding(decoder_inputs)
        decoder_embeddings = tf.squeeze(decoder_embeddings, axis=-2)

        transformer_output = self.transformer_decoder(decoder_embeddings, encoder_output, training=training)

        output = self.dense_output(transformer_output)
        return output

    def model_step(self, inputs, training=False):
        encoder_inputs, decoder_inputs = self.input_layer(inputs)
        output = self((encoder_inputs, decoder_inputs), training=training)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "batch_size": self.batch_size,
                "dropout": self.dropout,
                "dff": self.dff,
                "d_model": self.d_model,
                "enc_layers": self.enc_layers,
                "enc_heads": self.enc_heads,
                "max_bp": self.max_bp,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        base_model = tf.keras.saving.deserialize_keras_object(config.pop("base_model"))
        return cls(base_model=base_model, **config)



