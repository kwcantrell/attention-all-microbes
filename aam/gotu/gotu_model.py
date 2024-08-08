"""gotu_model.py"""

import biom
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm

from aam.callbacks import SaveModel
from aam.layers import InputLayer
from aam.nuc_model import BaseNucleotideModel
from aam.utils import LRDecrease
from aam.utils import float_mask


@tf.keras.saving.register_keras_serializable(package="GOTUModel")
class GOTUModel(BaseNucleotideModel):
    def __init__(
        self,
        base_model,
        dropout_rate,
        batch_size,
        max_bp,
        num_gotus,
        count_ff_dim=32,
        num_layers=2,
        num_attention_heads=8,
        dff=32,
        use_attention_loss=False,
        **kwargs,
    ):
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
        self.num_gotus = num_gotus
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
            intermediate_size=self.dff,
            norm_first=True,
            activation="relu",
        )

        self.dense_output = tf.keras.layers.Dense(num_gotus + 4)

    def build(self, input_shape=None):
        """
        simulate model execution using symbolic tensors
        """

        # input_seq = tf.keras.Input(
        #     shape=[None, self.max_bp],
        #     batch_size=self.batch_size,
        #     dtype=tf.int32,
        # )
        # input_rclr = tf.keras.Input(
        #     shape=[None],
        #     batch_size=self.batch_size,
        #     dtype=tf.float32,
        # )
        # input_gotu = tf.keras.Input(
        #     shape=[None, 1],
        #     batch_size=self.batch_size,
        #     dtype=tf.int32,
        # )
        # input_gotu_rclr = tf.keras.Input(
        #     shape=[None],
        #     batch_size=self.batch_size,
        #     dtype=tf.float32,
        # )

        # outputs = self.model_step((input_seq, input_rclr, input_gotu, input_gotu_rclr), training=False)
        # self.inputs = (input_seq, input_rclr, input_gotu, input_gotu_rclr)
        # self.outputs = outputs

        # super().build(
        #     (
        #         tf.TensorShape((None, None, 150)),
        #         tf.TensorShape((None, None)),
        #         tf.TensorShape((None, None, 1)),
        #         tf.TensorShape((None, None)),
        #     )
        # )
        self.built = True

    def make_call_function(self):
        """
        handles model exectution
        returns:
            a tuple of tf.tensors with first tensor being the "regression"
            component of type tf.float32 and the second representing the
            nucleotide sequences of type tf.int32
        """
        print("We hit the call function I swearsies")

        # @tf.function
        def one_step(inputs, training=False):
            print("model trace!", type(inputs), training)
            table_info, gotu_info = inputs
            pad_size = self.get_max_unique_asv(table_info)
            features, rclr = tf.map_fn(
                lambda x: self.get_table_data(x, pad_size, self.o_ids),
                table_info,
                fn_output_signature=(tf.string, tf.float32),
            )
            features = tf.cast(self.sequence_tokenizer(features), tf.int32)
            features = tf.stop_gradient(features)
            rclr = tf.stop_gradient(rclr)
            pad_size = self.get_max_unique_asv(gotu_info)
            gotu_features, gotu_rclr = tf.map_fn(
                lambda x: self.get_table_data(x, pad_size, self.gotu_ids),
                gotu_info,
                fn_output_signature=(tf.string, tf.float32),
            )
            gotu_features = tf.cast(self.gotu_tokenizer(gotu_features), tf.int32)
            gotu_features = tf.stop_gradient(gotu_features)
            gotu_rclr = tf.stop_gradient(gotu_rclr)
            tf.print("GOTU Features")
            tf.print(tf.shape(gotu_features))
            tf.print("ASV Features")
            tf.print(tf.shape(features))
            output = self.model_step((features, rclr, gotu_features, gotu_rclr), training=training)
            return output, gotu_features

        self.call_function = one_step

    def model_step(self, inputs, training=False):
        encoder_inputs, rclr, decoder_inputs, gotu_rclr = inputs
        encoder_output = self.base_model.feature_emb((encoder_inputs, rclr), return_nuc_attention=False, training=training)
        encoder_mask = tf.reduce_sum(encoder_inputs, axis=-1, keepdims=True)
        encoder_mask = tf.pad(encoder_mask, paddings=[[0, 0], [0, 1], [0, 0]], constant_values=1)
        encoder_mask = float_mask(encoder_mask)
        decoder_mask = float_mask(decoder_inputs)
        cross_attention_mask = tf.cast(tf.matmul(decoder_mask, encoder_mask, transpose_b=True), dtype=tf.bool)
        attention_mask = tf.cast(tf.matmul(decoder_mask, decoder_mask, transpose_b=True), dtype=tf.bool)

        decoder_embeddings = self.decoder_embedding(decoder_inputs)
        decoder_embeddings = tf.squeeze(decoder_embeddings, axis=-2)

        transformer_output = self.transformer_decoder(
            decoder_embeddings,
            encoder_output,
            self_attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            training=training,
        )

        output = self.dense_output(transformer_output)

        return output, output

    def _extract_data(self, data):
        (_, table_info), y = data
        pad_size = self.get_max_unique_asv(y)
        gotu_features, gotu_rclr = tf.map_fn(
            lambda x: self.get_table_data(x, pad_size, self.gotu_ids),
            y,
            fn_output_signature=(tf.string, tf.float32),
        )
        gotu_features = tf.cast(self.gotu_tokenizer(gotu_features), tf.int32)
        gotu_features = tf.stop_gradient(gotu_features)
        tf.print("Y shape PRE-SQUEEZE")
        tf.print(tf.shape(gotu_features))
        gotu_features = tf.squeeze(gotu_features, axis=-1)
        tf.print("Y shape Post-SQUEEZE")
        tf.print(tf.shape(gotu_features))
        return ([table_info, y], gotu_features)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "dropout_rate": self.dropout_rate,
                "batch_size": self.batch_size,
                "max_bp": self.max_bp,
                "num_gotus": self.num_gotus,
                "count_ff_dim": self.count_ff_dim,
                "num_layers": self.num_layers,
                "num_attention_heads": self.num_attention_heads,
                "dff": self.dff,
                "use_attention_loss": self.use_attention_loss,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        base_model = tf.keras.saving.deserialize_keras_object(config.pop("base_model"))
        return cls(base_model=base_model, **config)
