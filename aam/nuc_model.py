import tensorflow as tf
import tensorflow_models as tfm
from aam.losses import PairwiseLoss
from aam.layers import ReadHead, NucleotideEmbedding
from aam.metrics import PairwiseMAE, MAE
import abc


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
        sequence_tokenizer=None
):
    if include_count:
        model = NucModel(
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
    else:
        model = UnifracModel(
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
            sequence_tokenizer=sequence_tokenizer
        )

    return model


@tf.keras.saving.register_keras_serializable(package="UnifracModel")
class BaseNucleotideModel(tf.keras.Model):
    def __init__(
        self,
        batch_size,
        shift,
        scale,
        include_random=True,
        include_count=True,
        use_attention_loss=True,
        sequence_tokenizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.shift = shift
        self.scale = scale
        self.include_random = include_random
        self.include_count = include_count
        self.use_attention_loss = use_attention_loss
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.regresssion_loss = None
        self.sequence_tokenizer = sequence_tokenizer
        self.attention_loss = (
            tf.keras.losses.SparseCategoricalCrossentropy(
                ignore_class=0,
                from_logits=True,
                reduction="none"
            )
        )
        self.confidence_tracker = tf.keras.metrics.Mean(name="confidence")
        self.metric_traker = None
        self.make_call_function()

    @abc.abstractmethod
    def make_call_function(self):
        """
        handles model exectution
        returns:
            a tuple of tf.tensors with first tensor being the "regression"
            component of type tf.float32 and the second representing the
            nucleotide sequences of type tf.int64
        """
        return

    def call(self, inputs, training=None):
        return self.call_function(inputs, training=training)

    @abc.abstractmethod
    def build(self, input_shape):
        """
        simulate model execution using symbolic tensors
        """
        return

    def predict_step(self, data):
        raise NotImplementedError("Please Implement this method")

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
            loss = self.regresssion_loss(y, reg_out)
            if self.use_attention_loss:
                attention_loss = tf.reduce_mean(
                    self.attention_loss(
                        seq,
                        logits
                    ),
                    axis=-1
                )
                loss += attention_loss
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
        loss = self.regresssion_loss(y, reg_out)
        if self.use_attention_loss:
            attention_loss = tf.reduce_mean(
                self.attention_loss(
                    seq,
                    logits
                ),
                axis=-1
            )
            loss += attention_loss

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
            config['sequence_tokenizer'] = tf.keras.saving.serialize_keras_object(
                self.sequence_tokenizer
            )
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if 'sequence_tokenizer' in config:
            config['sequence_tokenizer'] = tf.keras.saving.deserialize_keras_object(
                config['sequence_tokenizer']
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
        dff,
        dropout_rate,
        use_attention_loss=False,
        **kwargs
    ):
        super().__init__(use_attention_loss=use_attention_loss, **kwargs)
        self.token_dim = token_dim
        self.max_bp = max_bp
        self.pca_hidden_dim = pca_hidden_dim
        self.pca_heads = pca_heads
        self.pca_layers = pca_layers
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.dff = dff
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
            dff,
            dropout_rate,
        )
        self.readhead = ReadHead(
            max_bp=max_bp,
            hidden_dim=pca_hidden_dim,
            num_heads=pca_heads,
            num_layers=pca_layers,
            output_dim=32
        )

    def make_call_function(self):
        """
        handles model exectution
        returns:
            a tuple of tf.tensors with first tensor being the "regression"
            component of type tf.float32 and the second representing the
            nucleotide sequences of type tf.int64
        """
        @tf.autograph.experimental.do_not_convert
        def one_step(inputs, training=None):
            ind, seq, rclr = inputs
            seq = self.sequence_tokenizer(seq)
            output, seq = self.feature_emb(
                (seq, rclr),
                training=training,
                include_random=self.include_random,
                include_count=self.include_count
            )
            output = self.readhead(output)
            return (output, seq)

        self.call_function = tf.function(one_step)

    def build(self, input_shape):
        input_seq = tf.keras.Input(
            shape=[None, self.max_bp],
            batch_size=self.batch_size,
            dtype=tf.int64
        )
        input_rclr = tf.keras.Input(
            shape=[None],
            batch_size=self.batch_size,
            dtype=tf.float32
        )
        output, _ = self.feature_emb(
            (input_seq, input_rclr),
            include_random=self.include_random,
            include_count=self.include_count
        )
        output = self.readhead(output)

    def _extract_data(self, data):
        x, y = data
        ind, seq, rclr = tf.nest.flatten(x)
        ind = tf.squeeze(ind)
        y = tf.gather(y, ind, axis=1, batch_dims=0)
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
            output_dim=1
        )

    def make_call_function(self):
        """
        handles model exectution
        returns:
            a tuple of tf.tensors with first tensor being the "regression"
            component of type tf.float32 and the second representing the
            nucleotide sequences of type tf.int64
        """
        @tf.autograph.experimental.do_not_convert
        def one_step(inputs, training=None):
            ind, seq, rclr = inputs
            output, seq = self.feature_emb((seq, rclr), training=training)
            output = self.readhead(output)
            return (output, seq)
        self.call_function = tf.function(one_step)

    def build(self, input_shape):
        input_seq = tf.keras.Input(
            shape=[None, self.max_bp],
            batch_size=self.batch_size,
            dtype=tf.int64
        )
        input_rclr = tf.keras.Input(
            shape=[None],
            batch_size=self.batch_size,
            dtype=tf.float32
        )
        output, _ = self.feature_emb((input_seq, input_rclr))
        output = self.readhead(output)

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
        shift,
        scale,
        use_attention_loss=False,
        **kwargs
    ):
        super().__init__(
            batch_size=base_model.batch_size,
            shift=shift,
            scale=scale,
            use_attention_loss=use_attention_loss,
            **kwargs
        )
        self.base_model = base_model
        self.base_model.feature_emb.trainable = False
        self.shift = shift
        self.scale = scale
        self.count_ff = tf.keras.Sequential([
            tf.keras.layers.Dense(
                1024,
                activation='relu',
                use_bias=True
            ),
            tf.keras.layers.Dense(
                self.base_model.pca_hidden_dim,
                use_bias=True
            ),
        ])
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(
                1024,
                activation='relu',
                use_bias=True
            ),
            tf.keras.layers.Dense(
                self.base_model.pca_hidden_dim,
                use_bias=True
            ),
            tf.keras.layers.LayerNormalization()
        ])
        self.attention_layer = tfm.nlp.models.TransformerEncoder(
            num_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
            dropout_rate=0.1,
            norm_first=True,
            activation='relu',
        )
        self.readhead = ReadHead(
            max_bp=base_model.max_bp,
            hidden_dim=base_model.pca_hidden_dim,
            num_heads=base_model.pca_heads,
            num_layers=base_model.pca_layers,
            output_dim=1
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.regresssion_loss = tf.keras.losses.MeanSquaredError()
        self.metric_traker = MAE(shift, scale, name="mae")

    def make_call_function(self):
        """
        handles model exectution
        returns:
            a tuple of tf.tensors with first tensor being the "regression"
            component of type tf.float32 and the second representing the
            nucleotide sequences of type tf.int64
        """
        @tf.autograph.experimental.do_not_convert
        def one_step(inputs, training=None):
            _, seq, rclr = inputs
            seq = self.base_model.sequence_tokenizer(seq)
            output, seq = self.base_model.feature_emb(
                (seq, rclr),
                training=training,
                include_random=self.include_random,
                include_count=self.base_model.include_count
            )
            if not self.base_model.include_count:
                rclr = tf.pad(
                    rclr,
                    paddings=[
                        [0, 0],
                        [0, tf.shape(output)[1] - tf.shape(rclr)[1]]
                    ],
                    constant_values=0
                )
                rclr = self.count_ff(
                    tf.expand_dims(
                        rclr,
                        axis=-1
                    )
                )
                # output = output + rclr
                rclr = tf.linalg.matmul(
                    output,
                    rclr,
                    transpose_b=True
                )
                rclr = tf.linalg.matmul(
                    rclr,
                    output,

                )
                output = self.ff(rclr)
            mask = tf.cast(
                tf.not_equal(
                    tf.pad(seq, [[0, 0], [0, 1], [0, 0]], constant_values=1),
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
            output = self.attention_layer(
                output,
                attention_mask=attention_mask,
                training=training
            )
            output = self.readhead(output)
            return (output, seq)
        self.call_function = tf.function(one_step)

    def build(self, input_shape):
        seq = tf.keras.Input(
            shape=[None, self.base_model.max_bp],
            batch_size=self.base_model.batch_size,
            dtype=tf.int64
        )
        rclr = tf.keras.Input(
            shape=[None],
            batch_size=self.base_model.batch_size,
            dtype=tf.float32
        )
        output, seq = self.base_model.feature_emb(
            (seq, rclr),
            include_random=self.include_random,
            include_count=self.base_model.include_count
        )
        if not self.base_model.include_count:
            rclr = tf.pad(
                rclr,
                paddings=[
                    [0, 0],
                    [0, tf.shape(output)[1] - tf.shape(rclr)[1]]
                ],
                constant_values=0
            )
            rclr = self.count_ff(
                tf.expand_dims(
                    rclr,
                    axis=-1
                )
            )
            # output = output + rclr
            rclr = tf.linalg.matmul(
                output,
                rclr,
                transpose_b=True
            )
            rclr = tf.linalg.matmul(
                rclr,
                output,
            )
            output = self.ff(rclr)
        output = self.attention_layer(output)
        output = self.readhead(output)

    def _extract_data(self, data):
        x, y = data
        ind, seq, rclr = tf.nest.flatten(x)
        ind = tf.squeeze(ind)
        return ((ind, seq, rclr), y)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "base_model": tf.keras.saving.serialize_keras_object(
                self.base_model
            ),
            "shift": self.shift,
            "scale": self.scale,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["base_model"] = tf.keras.saving.deserialize_keras_object(
            config["base_model"]
        )
