"""gotu_model.py"""

import biom
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm
from aam.callbacks import SaveModel
from aam.layers import InputLayer
from aam.nuc_model import BaseNucleotideModel
from aam.utils import LRDecrease, float_mask


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
        num_layers=4,
        num_attention_heads=8,
        dff=1024,
        use_attention_loss=False,
        start_token=None,
        end_token=None,
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
        self.use_attention_loss = False
        self.regresssion_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, ignore_class=0
        )
        self.start_token = start_token
        self.end_token = end_token
        self.metric_traker = tf.keras.metrics.Mean(name="loss")

        self.input_layer = InputLayer(name="input_layer")

        self.base_model = base_model
        self.base_model.trainable = False  # Freeze the base model

        self.decoder_embedding = tf.keras.layers.Embedding(
            input_dim=num_gotus + 4,
            output_dim=128,
            embeddings_initializer="uniform",
            input_length=1,
            name="decoder_embedding",
        )
        self.positional_encodings = tfm.nlp.layers.PositionEmbedding(
            max_length=1500, seq_axis=1
        )
        self.transformer_decoder = tfm.nlp.models.TransformerDecoder(
            num_layers=self.num_layers,
            dropout_rate=0.1,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.dff,
            norm_first=True,
            activation="relu",
            name="GOTU_Decoder",
        )

        self.dense_output = tf.keras.layers.Dense(num_gotus + 4, name="dense_output")
        self.linear_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)

    def build(self, input_shape=None):
        """
        simulate model execution using symbolic tensors
        """
        input_seq = tf.keras.Input(
            shape=[None, self.max_bp],
            batch_size=self.batch_size,
            dtype=tf.int32,
        )
        input_rclr = tf.keras.Input(
            shape=[None],
            batch_size=self.batch_size,
            dtype=tf.float32,
        )
        input_decoder = tf.keras.Input(
            shape=[None, 1], batch_size=self.batch_size, dtype=tf.int32
        )
        outputs = self.model_step(
            (input_seq, input_rclr, input_decoder), training=False
        )
        self.inputs = (input_seq, input_rclr, input_decoder)
        self.outputs = outputs

        # super().build(
        #     (
        #         tf.TensorShape((None, None, 150)),
        #         tf.TensorShape((None, None)),
        #         tf.TensorShape((None, None, 1)),
        #     )
        # )

    def make_call_function(self):
        """
        handles model exectution
        returns:
            a tuple of tf.tensors with first tensor being the "regression"
            component of type tf.float32 and the second representing the
            nucleotide sequences of type tf.int32
        """

        # @tf.function
        def one_step(inputs, training=False):
            print("model trace!", type(inputs), training)
            table_info, gotu_features = inputs
            pad_size = self.get_max_unique_asv(table_info)
            features, rclr = tf.map_fn(
                lambda x: self.get_table_data(x, pad_size, self.o_ids),
                table_info,
                fn_output_signature=(tf.string, tf.float32),
            )
            features = tf.cast(self.sequence_tokenizer(features), tf.int32)
            features = tf.stop_gradient(features)
            rclr = tf.stop_gradient(rclr)
            output = self.model_step((features, rclr, gotu_features), training=training)
            return output

        self.call_function = one_step

    def model_step(self, inputs, training=False):
        encoder_inputs, rclr, decoder_inputs = inputs

        encoder_output = self.base_model.feature_emb(
            (encoder_inputs, rclr), return_nuc_attention=False, training=training
        )
        encoder_mask = tf.reduce_sum(encoder_inputs, axis=-1, keepdims=True)
        encoder_mask = tf.pad(
            encoder_mask, paddings=[[0, 0], [0, 1], [0, 0]], constant_values=1
        )
        encoder_mask = float_mask(encoder_mask)

        decoder_inputs = tf.pad(
            decoder_inputs, [[0, 0], [1, 0], [0, 0]], constant_values=self.start_token
        )

        decoder_mask = float_mask(decoder_inputs)

        if self.trainable and training:
            random_mask = tf.random.uniform(
                tf.shape(decoder_inputs), minval=0, maxval=1, dtype=tf.float32
            )
            random_mask = tf.less_equal(random_mask, 0.9)
            random_mask = tf.cast(random_mask, dtype=tf.int32)
            decoder_inputs = tf.multiply(decoder_inputs, random_mask)
        cross_attention_mask = tf.cast(
            tf.matmul(decoder_mask, encoder_mask, transpose_b=True),
            dtype=self.compute_dtype,
        )
        attention_mask = tf.cast(
            tf.matmul(decoder_mask, decoder_mask, transpose_b=True),
            dtype=self.compute_dtype,
        )
        timestep_mask = tf.linalg.band_part(
            tf.ones_like(attention_mask, dtype=self.compute_dtype), -1, 0
        )
        attention_mask = tf.multiply(attention_mask, timestep_mask)
        decoder_embeddings = self.decoder_embedding(decoder_inputs)
        decoder_embeddings = tf.squeeze(decoder_embeddings, axis=-2)
        decoder_embeddings = decoder_embeddings + self.positional_encodings(
            decoder_embeddings
        )

        transformer_output = self.transformer_decoder(
            decoder_embeddings,
            encoder_output,
            self_attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            training=training,
        )
        transformer_output = transformer_output[:, :-1, :]
        output = self.dense_output(transformer_output)
        output = self.linear_activation(output)
        transformer_output_fake = self.linear_activation(
            transformer_output
        )  # cheezy hack, might remove later
        return (output, transformer_output_fake), output

    @tf.function
    def infer(self, inputs):
        encoder_output, encoder_mask, decoder_inputs = inputs
        decoder_inputs = tf.expand_dims(decoder_inputs, axis=-1)
        # tf.print("ENCODER INPUTS", encoder_inputs.shape)
        # tf.print("DECODER_INPUTS", decoder_inputs.shape)
        decoder_mask = float_mask(decoder_inputs)
        cross_attention_mask = tf.cast(
            tf.matmul(decoder_mask, encoder_mask, transpose_b=True),
            dtype=self.compute_dtype,
        )
        attention_mask = tf.cast(
            tf.matmul(decoder_mask, decoder_mask, transpose_b=True),
            dtype=self.compute_dtype,
        )
        timestep_mask = tf.linalg.band_part(
            tf.ones_like(attention_mask, dtype=self.compute_dtype), -1, 0
        )
        attention_mask = tf.multiply(attention_mask, timestep_mask)
        decoder_embeddings = self.decoder_embedding(decoder_inputs)
        # tf.print(
        #     "DECODER EMBEDDINGS", decoder_embeddings.shape, "\n", decoder_embeddings
        # )
        decoder_embeddings = tf.squeeze(decoder_embeddings, axis=-2)
        decoder_embeddings = decoder_embeddings + self.positional_encodings(
            decoder_embeddings
        )
        transformer_output = self.transformer_decoder(
            decoder_embeddings,
            encoder_output,
            self_attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
        )
        output = transformer_output[:, -1, :]
        output = self.dense_output(output)
        # output = self.linear_activation(output)
        # transformer_output_fake = self.linear_activation(
        #     transformer_output
        # )  # cheezy hack, might remove later
        # tf.print("OUTPUT", output.shape)

        return output

    def _extract_data(self, data):
        _, table_info, gotu_features = data

        return (table_info, gotu_features), gotu_features

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
                "start_token": self.start_token,
                "end_token": self.end_token,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        base_model = tf.keras.saving.deserialize_keras_object(config.pop("base_model"))
        return cls(base_model=base_model, **config)
