import abc
from typing import Union

import tensorflow as tf
from aam.metrics import MAE

from attention_regression.layers import FeatureEmbedding, ReadHead


def _construct_regressor(
    batch_size: int,
    dropout_rate: float,
    pca_hidden_dim: int,
    pca_heads: int,
    pca_layers: int,
    dff: int,
    token_dim: int,
    ff_clr,
    attention_layers: int,
    attention_heads: int,
    output_dim: int,
    max_bp: int,
    shift=None,
    scale=None,
    include_count=True,
    include_random=True,
    o_ids=None,
    sequence_tokenizer=None,
):
    if include_count:
        model = AtttentionRegression(
            pca_hidden_dim,
            max_bp,
            pca_hidden_dim,
            pca_heads,
            pca_layers,
            attention_heads,
            attention_layers,
            dff,
            dropout_rate,
            batch_size=batch_size,
            shift=shift,
            scale=scale,
            include_random=include_random,
            include_count=include_count,
        )
    return model


def cast_to_loss_shape(tensor):
    return tf.reshape(tensor, shape=(-1, 1))


@tf.keras.saving.register_keras_serializable(package="BaseModel")
class BaseModel(tf.keras.Model):
    def __init__(
        self,
        batch_size,
        shift=0,
        scale=1,
        include_random=True,
        include_count=True,
        seq_mask_rate=0.1,
        use_attention_loss=True,
        o_ids=None,
        sequence_tokenizer: Union[tf.keras.layers.StringLookup, None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.shift = shift
        self.scale = scale
        self.include_random = include_random
        self.include_count = include_count
        self.seq_mask_rate = seq_mask_rate
        self.use_attention_loss = use_attention_loss
        self.o_ids = o_ids
        self.sequence_tokenizer = sequence_tokenizer
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.regresssion_loss = None
        self.attention_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            ignore_class=0, from_logits=True, reduction="none"
        )
        self.confidence_tracker = tf.keras.metrics.Mean(name="confidence")
        self.metric_traker = None
        self.make_call_function()

    @abc.abstractmethod
    def model_step(self, inputs, training=None):
        return

    def get_table_data(self, sparse_row):
        # count
        dense = tf.cast(sparse_row.values, dtype=tf.float32)
        dense = tf.scatter_nd(
            sparse_row.indices, dense, shape=[sparse_row.dense_shape[0]]
        )
        return dense

    def make_call_function(self):
        """
        handles model exectution
        returns:
            a tuple of tf.tensors with first tensor being the "regression"
            component of type tf.float32 and the second representing the
            nucleotide sequences of type tf.int64
        """

        # @tf.autograph.experimental.do_not_convert
        def one_step(inputs, training=None):
            table_info = inputs
            counts = tf.map_fn(
                self.get_table_data, table_info, fn_output_signature=tf.float32
            )
            output = self.model_step(
                (
                    tf.stop_gradient(table_info.indices),
                    tf.stop_gradient(tf.cast(table_info.indices[:, 1], dtype=tf.int32)),
                    tf.stop_gradient(counts),
                ),
                training=training,
            )
            # return (output, features)
            return output

        self.call_function = one_step

    def call(self, inputs, training=False):
        return self.call_function(inputs, training=training)

    def build(self, input_shape=None):
        """
        simulate model execution using symbolic tensors
        """
        input_ind = tf.keras.Input(shape=[None, 2], dtype=tf.int32)
        input_seq = tf.keras.Input(shape=[None], dtype=tf.int32)
        input_rclr = tf.keras.Input(shape=[None], dtype=tf.float32)
        self.model_step((input_ind, input_seq, input_rclr))

    def compile(self, o_ids=None, **kwargs):
        super().compile(**kwargs)
        if o_ids is not None:
            self.o_ids = o_ids

    def predict_step(self, data):
        (sample_indices, table_info), y = data

        counts = tf.map_fn(
            self.get_table_data, table_info, fn_output_signature=tf.float32
        )
        output = self.model_step(
            (
                tf.stop_gradient(table_info.indices),
                tf.stop_gradient(tf.cast(table_info.indices[:, 1], dtype=tf.int32)),
                tf.stop_gradient(counts),
            ),
            training=False,
        )
        return output

    @abc.abstractmethod
    def _extract_data(self, data):
        """
        this should become a tf.keras.layer for better preprosses with tf.data

        returns:
            inputs: a tuple of tf.tensors representing model input
            y: a tf.tensor
        """
        return

    def train_step(self, data):
        """
        Need to add checks to verify input/output tuples are formatted correctly
        """
        inputs, y_ad = self._extract_data(data)
        # y_ad = cast_to_loss_shape(y_ad)
        with tf.GradientTape() as tape:
            # Forward pass
            reg_out = self(inputs, training=True)
            # reg_out = cast_to_loss_shape(reg_out)

            # Compute regression loss
            loss = self.regresssion_loss(y_ad, reg_out)
            model_reg = tf.reduce_sum(self.losses)
            loss += model_reg
            self.loss_tracker.update_state(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.metric_traker.update_state(y_ad, reg_out)
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.metric_traker.result(),
        }

    def test_step(self, data):
        """
        Need to add checks to verify input/output tuples are formatted correctly
        """
        inputs, y_ad = self._extract_data(data)
        # y_ad = cast_to_loss_shape(y_ad)
        # Forward pass
        reg_out = self(inputs, training=False)
        # reg_out = cast_to_loss_shape(reg_out)

        # Compute regression loss
        loss = self.regresssion_loss(y_ad, reg_out)
        self.loss_tracker.update_state(loss)

        # Compute our own metrics
        self.metric_traker.update_state(y_ad, reg_out)
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.metric_traker.result(),
        }

    def get_config(self):
        base_config = super().get_config()
        config = {
            "batch_size": self.batch_size,
            "shift": self.shift,
            "scale": self.scale,
            "include_random": self.include_random,
            "include_count": self.include_count,
            "use_attention_loss": self.use_attention_loss,
        }
        if self.sequence_tokenizer is not None:
            config["sequence_tokenizer"] = tf.keras.saving.serialize_keras_object(
                self.sequence_tokenizer
            )
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if "sequence_tokenizer" in config:
            config["sequence_tokenizer"] = tf.keras.saving.deserialize_keras_object(
                config["sequence_tokenizer"]
            )
        return cls(**config)


@tf.keras.saving.register_keras_serializable(package="AtttentionRegression")
class AtttentionRegression(BaseModel):
    def __init__(
        self,
        batch_size,
        token_dim,
        d_model,
        attention_heads,
        attention_layers,
        dff,
        dropout_rate,
        use_attention_loss=False,
        **kwargs,
    ):
        super().__init__(batch_size, use_attention_loss=use_attention_loss, **kwargs)
        self.token_dim = token_dim
        self.d_model = d_model
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        # self.regresssion_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # self.metric_traker = tf.keras.metrics.BinaryAccuracy()
        self.regresssion_loss = tf.keras.losses.MeanSquaredError()
        self.metric_traker = MAE(self.shift, self.scale, name="mae")
        self.feature_emb: FeatureEmbedding = FeatureEmbedding(
            len(self.sequence_tokenizer.get_vocabulary()),
            token_dim,
            d_model,
            attention_heads,
            attention_layers,
            dff,
            dropout_rate,
        )
        self.readhead = ReadHead(output_dim=1)

    def model_predict_step(self, inputs, training=None):
        output = self.feature_emb.process_seq(inputs, training=False)
        output = tf.argmax(self.readhead(output), axis=-1)
        return output

    def model_step(self, inputs, training=None):
        output = self.feature_emb(inputs, training=training, batch_size=self.batch_size)
        output = self.readhead(output)
        return output

    def _extract_data(self, data):
        (sample_indices, table_info), y = data
        return (table_info, y)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "token_dim": self.token_dim,
            "d_model": self.d_model,
            "attention_heads": self.attention_heads,
            "attention_layers": self.attention_layers,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        }
        return {**base_config, **config}
