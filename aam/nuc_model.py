import abc

import tensorflow as tf

from aam.layers import (
    InputLayer,
    NucleotideEmbedding,
    ReadHead,
    TransferAttention,
)
from aam.losses import PairwiseLoss
from aam.types import FloatTensor
from aam.utils import float_mask


def shifted_log_transform(count_tensor: FloatTensor) -> FloatTensor:
    """Computes shifted (natural) logarithm transformation log(x + 1)
    element-wise
    See https://doi.org/10.1038/s41592-023-01814-1

    Args:
        count_tensor: A FloatTensor

    Returns:
        A FloatTensor
    """
    # log_counts = tf.math.log1p(count_tensor)
    closure = count_tensor / tf.reduce_sum(count_tensor, axis=-1, keepdims=True)
    log_portions = tf.where(closure > 0, tf.math.log(closure), closure)
    return closure


@tf.keras.saving.register_keras_serializable(package="BaseNucleotideModel")
class BaseNucleotideModel(tf.keras.Model):
    def __init__(
        self,
        batch_size,
        shift=0,
        scale=1,
        include_random=False,
        include_count=False,
        seq_mask_rate=0.1,
        use_attention_loss=True,
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
        self.sequence_tokenizer = sequence_tokenizer
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.regresssion_loss = None
        self.attention_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction="none")
        self.confidence_tracker = tf.keras.metrics.Mean(name="confidence")
        self.metric_traker = None
        self.output_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        self.counter = self.add_weight("counter", initializer="zeros", trainable=False, dtype=tf.float32)

        def shift_and_scale(scale, shift):
            def _inner(tensor):
                return tensor * scale + shift

            return _inner

        self.shift_and_scale = tf.function(shift_and_scale(self.scale, self.shift))
        self.make_call_function()

    @abc.abstractmethod
    def _construct_model(self):
        pass

    @abc.abstractmethod
    def model_step(self, inputs, training=False):
        pass

    def get_max_unique_asv(self, sparse_tensor):
        features_per_sample = tf.sparse.reduce_sum(tf.sparse.map_values(tf.ones_like, sparse_tensor), axis=-1)
        max_features = tf.reduce_max(features_per_sample)
        return tf.cast(max_features, dtype=tf.int32)

    def get_table_data(self, sparse_row, pad_size, o_ids):
        # count
        dense = shifted_log_transform(tf.cast(sparse_row.values, dtype=tf.float32))
        non_zero = tf.cast(tf.shape(dense)[0], dtype=tf.int32)
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
            nucleotide sequences of type tf.int32
        """

        # @tf.function
        def one_step(inputs, training=False):
            print("model trace!", type(inputs), training)
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

        self.call_function = one_step

    def call(self, inputs, training=False):
        return self.call_function(inputs, training=training)

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
        outputs = self.model_step((input_seq, input_rclr), training=False)
        self.inputs = (input_seq, input_rclr)
        self.outputs = outputs

        super().build((tf.TensorShape((None, None, 150)), tf.TensorShape((None, None))))

    def compile(
        self,
        o_ids=None,
        gotu_ids=None,
        sequence_tokenizer=None,
        gotu_tokenizer=None,
        optimizer="rmsprop",
        polyval=None,
        **kwargs,
    ):
        if sequence_tokenizer is not None:
            self.sequence_tokenizer = sequence_tokenizer

        if gotu_ids is not None:
            self.gotu_ids = gotu_ids

        if gotu_tokenizer is not None:
            self.gotu_tokenizer = gotu_tokenizer

        if polyval is not None:
            self.feature_emb.pca_layer.polyval = polyval

        if o_ids is not None:
            self.o_ids = o_ids
            self.make_call_function()
        super().compile(optimizer=optimizer, **kwargs)

    def predict_step(self, data):
        inputs, y = self._extract_data(data)
        # Forward pass
        outputs, _ = self(inputs, training=False)
        reg_out, _ = tf.nest.flatten(outputs, expand_composites=True)
        reg_out = self.shift_and_scale(reg_out)
        y = self.shift_and_scale(y)
        return tf.squeeze(reg_out), tf.squeeze(y)

    @abc.abstractmethod
    def _extract_data(self, data) -> tuple[int, int]:
        """
        this should become a tf.keras.layer for better preprosses with tf.data

        returns:
            inputs: a tuple of tf.tensors representing model input
            y: a tf.tensor
        """
        pass

    def regularization_loss(self):
        return self.losses

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
            reg_loss = tf.reduce_mean(self.regresssion_loss(y, reg_out))
            loss = reg_loss
            if self.use_attention_loss:
                seq_cat = tf.one_hot(seq, depth=6)  # san -> san6

                asv_mask = float_mask(tf.reduce_sum(seq, axis=-1, keepdims=True))  # san -> sa1

                asv_counts = float_mask(tf.reduce_sum(seq, axis=-1))  # san -> sa
                asv_counts = asv_counts = tf.reduce_sum(asv_counts, axis=-1, keepdims=True)  # sa -> s1)

                # nucleotide level
                asv_loss = self.attention_loss(seq_cat, logits)  # san6 -> san
                asv_loss = asv_loss * asv_mask
                asv_loss = tf.reduce_sum(asv_loss, axis=-1)  # san -> sa

                # asv_level
                asv_loss = tf.reduce_sum(asv_loss, axis=-1, keepdims=True)  # sa -> s1
                asv_loss = asv_loss / asv_counts
                # tf.print(tf.shape(loss), tf.shape(asv_loss))

                # total
                # asv_loss = tf.reduce_sum(asv_loss)
                loss = loss + tf.reduce_mean(asv_loss)
            # loss = tf.squeeze(loss, axis=-1)
            # loss = tf.reduce_mean(loss)
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        # # Compute gradients
        # trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)

        # # Update weights
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.metric_traker.update_state(reg_loss)
        if self.use_attention_loss:
            self.confidence_tracker.update_state(asv_loss)
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
        reg_loss = tf.reduce_mean(self.regresssion_loss(y, reg_out))
        loss = reg_loss
        if self.use_attention_loss:
            seq_cat = tf.one_hot(seq, depth=6)  # san -> san6

            asv_mask = float_mask(tf.reduce_sum(seq, axis=-1, keepdims=True))  # san -> sa1

            asv_counts = float_mask(tf.reduce_sum(seq, axis=-1))  # san -> sa
            asv_counts = asv_counts = tf.reduce_sum(asv_counts, axis=-1, keepdims=True)  # sa -> s1)

            # nucleotide level
            asv_loss = self.attention_loss(seq_cat, logits)  # san6 -> san
            asv_loss = asv_loss * asv_mask
            asv_loss = tf.reduce_sum(asv_loss, axis=-1)  # san -> sa

            # asv_level
            asv_loss = tf.reduce_sum(asv_loss, axis=-1, keepdims=True)  # sa -> s1
            asv_loss = asv_loss / asv_counts
            # tf.print(tf.shape(loss), tf.shape(asv_loss))

            # total
            loss = loss + tf.reduce_mean(asv_loss)
        # loss = tf.squeeze(loss, axis=-1)
        # loss = tf.reduce_mean(loss)
        self.loss_tracker.update_state(loss)

        # Compute our own metrics
        self.metric_traker.update_state(reg_loss)
        if self.use_attention_loss:
            self.confidence_tracker.update_state(asv_loss)
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

    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        expand_nested=False,
        show_trainable=True,
        layer_range=None,
    ):
        # if layer_range is None:
        #     layer_range = self.layer_range

        return super().summary(
            line_length,
            positions,
            print_fn,
            expand_nested,
            show_trainable,
            layer_range,
        )

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
            config["sequence_tokenizer"] = tf.keras.saving.serialize_keras_object(self.sequence_tokenizer)
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if "sequence_tokenizer" in config:
            config["sequence_tokenizer"] = tf.keras.saving.deserialize_keras_object(config["sequence_tokenizer"])
        model = cls(**config)
        # model.compile()
        model.build()
        return model


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
        polyval=1.0,
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
        self.polyval = polyval
        self.regresssion_loss = PairwiseLoss()
        self.metric_traker = tf.keras.metrics.Mean(name="loss")

        # layers used in model
        self.input_layer = InputLayer(name="unifrac_input")
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
            polyval,
        )
        self.readhead = ReadHead(
            max_bp=max_bp,
            hidden_dim=pca_hidden_dim,
            num_heads=pca_heads,
            num_layers=pca_layers,
            output_dim=32,
            output_nuc=use_attention_loss,
            name="unifrac_output",
        )
        self.linear_activation = tf.keras.layers.Activation("linear", dtype=tf.float32)
        self.layer_range = ["unifrac_input", "unifrac_output"]

    def model_step(self, inputs, training=False):
        inputs = self.input_layer(inputs)
        output, nuc_attention = self.feature_emb(
            inputs,
            training=training,
        )
        (reg_out, att_out) = self.readhead((output, nuc_attention), training=training)
        reg_out = self.linear_activation(reg_out)
        att_out = self.linear_activation(att_out)
        # tf.print(reg_out)
        return (reg_out, att_out)

    def _extract_data(self, data):
        (sample_indices, table_info), y = data
        sample_indices = tf.squeeze(sample_indices)
        y = tf.gather(y, sample_indices, axis=1, batch_dims=0)
        return (table_info, y)

    @tf.function
    def sequence_embedding(self, seq, squeeze=True):
        """returns sequenuce embeddings

        Args:
            seq (StringTensor): ASV sequences.

        Returns:
            _type_: _description_
        """
        seq = tf.cast(self.sequence_tokenizer(seq), tf.int32)
        seq = tf.expand_dims(seq, axis=0)
        sequence_embedding = self.feature_emb.sequence_embedding(seq)
        if squeeze:
            return tf.squeeze(sequence_embedding, axis=0)
        else:
            return sequence_embedding

    def edit_distance(self, seq):
        """computes the edit distance between true seq and AM^-1

        Args:
            seq (StringTensor): ASV sequences.

        Returns:
            _type_: _description_
        """
        seq = tf.cast(self.sequence_tokenizer(seq), tf.int32)
        seq = tf.expand_dims(seq, axis=0)
        sequence_embedding = self.feature_emb.sequence_embedding(seq)

        sequence_logits = self.readhead.sequence_logits(sequence_embedding)
        pred_seq = tf.argmax(tf.nn.softmax(sequence_logits, axis=-1), axis=-1, output_type=tf.int32)
        sequence_mismatch_tokens = tf.not_equal(pred_seq, seq)
        sequence_mismatch_counts = tf.reduce_sum(tf.cast(sequence_mismatch_tokens, dtype=tf.int32), axis=-1)
        mask = tf.reshape(tf.not_equal(seq[:, :, 0], 0), shape=(-1,))
        return tf.reshape(sequence_mismatch_counts, shape=(-1,))[mask]

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
            "polyval": self.polyval,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(package="TransferLearnNucleotideModel")
class TransferLearnNucleotideModel(BaseNucleotideModel):
    def __init__(
        self,
        base_model,
        dropout_rate,
        batch_size,
        max_bp,
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
        self.regresssion_loss = tf.keras.losses.MeanSquaredError(reduction="none")
        self.metric_traker = tf.keras.metrics.Mean(name="loss")

        # layers used in model
        self.input_layer = InputLayer(name="transfer_input")
        self.base_model = base_model
        self.base_model.trainable = False
        self.attention_layer = TransferAttention(self.dropout_rate, d_model=d_model, name="transfer_output")
        self.layer_range = ["transfer_input", "transfer_output"]

    def model_step(self, inputs, training=False):
        inputs = self.input_layer(inputs)
        seq, rel_count = inputs
        embeddings = self.base_model.feature_emb(inputs, return_attention=True)

        embeddings = self.attention_layer((embeddings, rel_count), training=training)
        output = self.readhead(embeddings, training=training)
        return output

    def _extract_data(self, data):
        (_, table_info), y = data
        return (table_info, y)

    def regularization_loss(self):
        uni_losses = len(self.get_layer("unifrac_model").losses)
        return self.losses[uni_losses:]

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
