import abc

import tensorflow as tf
import tensorflow_models as tfm

from aam.layers import NucleotideEmbedding, ReadHead
from aam.losses import PairwiseLoss
from aam.metrics import MAE, PairwiseMAE
from aam.utils import float_mask


@tf.keras.saving.register_keras_serializable(package="BaseNucleotideModel")
class BaseNucleotideModel(tf.keras.Model):
    def __init__(
        self,
        batch_size,
        shift=0,
        scale=1,
        include_random=False,
        include_count=True,
        seq_mask_rate=0.1,
        use_attention_loss=True,
        o_ids=None,
        sequence_tokenizer=None,
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

    def get_max_unique_asv(self, sparse_tensor):
        max_features = tf.reduce_max(
            tf.reduce_sum(
                tf.sparse.to_dense(
                    tf.sparse.map_values(
                        lambda x: tf.ones_like(x, dtype=tf.int32), sparse_tensor
                    )
                ),
                axis=-1,
            )
        )
        return max_features // 32 * 32 + 32

    def get_table_data(self, sparse_row, pad_size, o_ids):
        # count
        dense = tf.cast(sparse_row.values, dtype=tf.float32)
        gx = tf.exp(tf.reduce_mean(tf.math.log(dense)))
        dense = tf.math.log(dense / gx)
        non_zero = tf.shape(dense)[0]
        zeros = tf.zeros(pad_size - non_zero, dtype=tf.float32)
        count_info = tf.concat([dense, zeros], axis=0)

        # ids
        features = tf.gather(o_ids, sparse_row.indices)
        masks = tf.zeros((pad_size - non_zero, 1), dtype=tf.string)
        feature_info = tf.concat([features, masks], axis=0)
        return (feature_info, count_info)

    def make_call_function(self):
        """
        handles model exectution
        returns:
            a tuple of tf.tensors with first tensor being the "regression"
            component of type tf.float32 and the second representing the
            nucleotide sequences of type tf.int64
        """

        def one_step(inputs, training=None):
            table_info = inputs
            pad_size = self.get_max_unique_asv(table_info)
            features, rclr = tf.map_fn(
                lambda x: self.get_table_data(x, pad_size, self.o_ids),
                table_info,
                fn_output_signature=(tf.string, tf.float32),
            )
            features = tf.cast(self.sequence_tokenizer(features), tf.int32)
            features = tf.stop_gradient(features)
            rclr = tf.stop_gradient(rclr)
            output = self.model_step((features, rclr), training=training)
            return (output, features)

        self.call_function = tf.function(one_step)

    def call(self, inputs, training=False):
        return self.call_function(inputs, training=training)

    def build(self, input_shape=None):
        """
        simulate model execution using symbolic tensors
        """
        input_seq = tf.keras.Input(shape=[None, self.max_bp], dtype=tf.int32)
        input_rclr = tf.keras.Input(shape=[None], dtype=tf.float32)
        self.model_step((input_seq, input_rclr))

    def compile(self, o_ids=None, **kwargs):
        super().compile(**kwargs)
        if o_ids is not None:
            self.o_ids = o_ids

    def predict_step(self, data):
        inputs, _ = self._extract_data(data)
        # Forward pass
        outputs, _ = self(inputs, training=False)
        reg_out, _ = tf.nest.flatten(outputs, expand_composites=True)
        return reg_out

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
        inputs, y = self._extract_data(data)
        with tf.GradientTape() as tape:
            # Forward pass
            outputs, seq = self(inputs, training=True)
            reg_out, logits = tf.nest.flatten(outputs, expand_composites=True)

            # Compute regression loss
            loss = tf.squeeze(self.regresssion_loss(y, reg_out))
            # TODO: Abstract to function for variable loss steps in training
            if self.use_attention_loss:
                counts = tf.reduce_sum(
                    float_mask(seq),
                    axis=-1,
                )
                attention_loss = tf.reduce_sum(
                    self.attention_loss(seq, logits), axis=-1
                )
                attention_loss = tf.reduce_sum(
                    tf.math.divide_no_nan(attention_loss, counts), axis=-1
                )
                loss = tf.reduce_mean(loss + attention_loss)
            loss += self.losses
            self.loss_tracker.update_state(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.metric_traker.update_state(y, reg_out)
        if self.use_attention_loss:
            self.confidence_tracker.update_state(attention_loss)
            return {
                "loss": self.loss_tracker.result(),
                "confidence": self.confidence_tracker.result(),
                "mae": self.metric_traker.result(),
            }
        else:
            return {
                "loss": self.loss_tracker.result(),
                "mae": self.metric_traker.result(),
            }

    def test_step(self, data):
        """
        Need to add checks to verify input/output tuples are formatted correctly
        """
        inputs, y = self._extract_data(data)

        # Forward pass
        outputs, seq = self(inputs, training=False)
        reg_out, logits = tf.nest.flatten(outputs, expand_composites=True)

        # Compute regression loss
        loss = tf.squeeze(self.regresssion_loss(y, reg_out))
        if self.use_attention_loss:
            counts = tf.reduce_sum(
                float_mask(seq),
                axis=-1,
            )
            attention_loss = tf.reduce_sum(self.attention_loss(seq, logits), axis=-1)
            attention_loss = tf.reduce_sum(
                tf.math.divide_no_nan(attention_loss, counts), axis=-1
            )
            loss = tf.reduce_mean(loss + attention_loss)
        self.loss_tracker.update_state(loss)

        # Compute our own metrics
        self.metric_traker.update_state(y, reg_out)
        if self.use_attention_loss:
            self.confidence_tracker.update_state(attention_loss)
            return {
                "loss": self.loss_tracker.result(),
                "confidence": self.confidence_tracker.result(),
                "mae": self.metric_traker.result(),
            }
        else:
            return {
                "loss": self.loss_tracker.result(),
                "mae": self.metric_traker.result(),
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


@tf.keras.saving.register_keras_serializable(package="UnifracModel")
class UnifracModel(BaseNucleotideModel):
    def __init__(
        self,
        token_dim,
        max_bp,
        pca_hidden_dim,
        pca_heads,
        pca_layers,
        attention_heads,
        attention_layers,
        attention_ff,
        dropout_rate,
        use_attention_loss=True,
        **kwargs,
    ):
        super().__init__(use_attention_loss=use_attention_loss, **kwargs)
        self.token_dim = token_dim
        self.max_bp = max_bp
        self.pca_hidden_dim = pca_hidden_dim
        self.pca_heads = pca_heads
        self.pca_layers = pca_layers
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.attention_ff = attention_ff
        self.dropout_rate = dropout_rate
        self.regresssion_loss = PairwiseLoss()
        self.metric_traker = PairwiseMAE()
        self.feature_emb = NucleotideEmbedding(
            token_dim,
            max_bp,
            pca_hidden_dim,
            pca_heads,
            pca_layers,
            attention_heads,
            attention_layers,
            attention_ff,
            dropout_rate,
        )
        self.readhead = ReadHead(
            max_bp=max_bp,
            hidden_dim=pca_hidden_dim,
            num_heads=pca_heads,
            num_layers=pca_layers,
            output_dim=32,
        )

    def model_step(self, inputs, training=False):
        output = self.feature_emb(
            inputs,
            training=training,
            include_random=self.include_random,
            seq_mask_rate=self.seq_mask_rate,
        )
        output = self.readhead(output)
        return output

    def _extract_data(self, data):
        (sample_indices, table_info), y = data
        sample_indices = tf.squeeze(sample_indices)
        y = tf.gather(y, sample_indices, axis=1, batch_dims=0)
        return (table_info, y)

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
            "attention_ff": self.attention_ff,
            "dropout_rate": self.dropout_rate,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(package="NucModel")
class NucModel(BaseNucleotideModel):
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
        **kwargs,
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
        self.regresssion_loss = tf.keras.losses.MeanSquaredError()
        self.metric_traker = MAE(self.shift, self.scale, name="mae")
        self.feature_emb = NucleotideEmbedding(
            token_dim,
            max_bp,
            pca_hidden_dim,
            pca_heads,
            pca_layers,
            attention_heads,
            attention_layers,
            dff,
            dropout_rate,
        )
        self.readhead = ReadHead(
            max_bp=max_bp,
            hidden_dim=pca_hidden_dim,
            num_heads=pca_heads,
            num_layers=pca_layers,
            output_dim=1,
        )

    def model_step(self, inputs, training=None):
        output, seq = self.feature_emb(inputs, training=training)
        output = self.readhead(output)
        return (output, seq)

    def _extract_data(self, data):
        x, y = data
        ind, seq, rclr = tf.nest.flatten(x)
        ind = tf.squeeze(ind)
        return ((ind, seq, rclr), y)

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


@tf.keras.saving.register_keras_serializable(package="TransferLearnNucleotideModel")
class TransferLearnNucleotideModel(BaseNucleotideModel):
    def __init__(
        self,
        base_model,
        dropout_rate,
        count_ff_dim=32,
        num_layers=2,
        num_attention_heads=8,
        dff=32,
        use_attention_loss=True,
        **kwargs,
    ):
        super().__init__(
            batch_size=base_model.batch_size,
            use_attention_loss=use_attention_loss,
            **kwargs,
        )
        self.base_model = base_model
        self.base_model.trainable = False
        self.max_bp = self.base_model.max_bp
        self.dropout_rate = dropout_rate
        self.count_ff_dim = count_ff_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.dff = dff
        self.sequence_tokenizer = self.base_model.sequence_tokenizer
        self.count_ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(count_ff_dim, activation="relu", use_bias=True),
                tf.keras.layers.Dense(self.base_model.pca_hidden_dim, use_bias=True),
                tf.keras.layers.LayerNormalization(),
            ]
        )
        self.attention_layer = tfm.nlp.models.TransformerEncoder(
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=dff,
            dropout_rate=dropout_rate,
            norm_first=True,
            activation="relu",
        )
        self.readhead = ReadHead(
            max_bp=base_model.max_bp,
            hidden_dim=base_model.pca_hidden_dim,
            num_heads=base_model.pca_heads,
            num_layers=base_model.pca_layers,
            output_dim=1,
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.regresssion_loss = tf.keras.losses.MeanSquaredError(reduction="none")
        self.metric_traker = MAE(self.shift, self.scale, name="mae")

    def model_step(self, inputs, training=None):
        seq, rclr = inputs
        base_emb = self.base_model.feature_emb
        output = base_emb(
            (seq, rclr),
            training=training,
            include_random=self.include_random,
            seq_mask_rate=self.seq_mask_rate,
        )
        output = tf.stop_gradient(output)
        seq, rclr = base_emb.add_seq_and_count_pad(seq, rclr)
        if not self.base_model.include_count:
            rclr = self.count_ff(tf.expand_dims(rclr, axis=-1))
            output = output + rclr

        mask = float_mask(seq)
        attention_mask = tf.cast(tf.matmul(mask, mask, transpose_b=True), dtype=tf.bool)
        output = self.attention_layer(
            output, attention_mask=attention_mask, training=training
        )
        output = self.readhead(output)
        return output

    def _extract_data(self, data):
        (_, table_info), y = data
        return (table_info, y)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
            "dropout_rate": self.dropout_rate,
            "count_ff_dim": self.count_ff_dim,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "dff": self.dff,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["base_model"] = tf.keras.saving.deserialize_keras_object(
            config["base_model"]
        )
