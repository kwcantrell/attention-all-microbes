import tensorflow as tf
import tensorflow_models as tfm
from aam.common.losses import PairwiseLoss
from aam.layers import ReadHead, NucleotideEmbedding
from aam.common.metrics import PairwiseMAE, MAE



def _construct_base(
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
        scale=None
):
    input = tf.keras.Input(
        shape=[None, max_bp],
        batch_size=batch_size,
        dtype=tf.int64
    )
    nuc_emb = NucleotideEmbedding(
        pca_hidden_dim,
        max_bp,
        pca_hidden_dim,
        pca_heads,
        pca_layers,
        attention_heads,
        attention_layers,
        dff,
        dropout_rate
    )
    output = nuc_emb(input)
    readhead = ReadHead(
        hidden_dim=pca_hidden_dim,
        num_heads=pca_heads,
        num_layers=pca_layers,
        output_dim=output_dim
    )
    output = readhead(output)
    model = UnifracModel(
        max_bp,
        batch_size,
        nuc_emb,
        readhead
    )
    # model = NucModel(
    #     max_bp,
    #     batch_size,
    #     nuc_emb,
    #     readhead,
        
    # )
    return model


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
        scale=None
):
    input = tf.keras.Input(
        shape=[None, max_bp],
        batch_size=batch_size,
        dtype=tf.int64
    )
    nuc_emb = NucleotideEmbedding(
        pca_hidden_dim,
        max_bp,
        pca_hidden_dim,
        pca_heads,
        pca_layers,
        attention_heads,
        attention_layers,
        dff,
        dropout_rate
    )
    # output = nuc_emb(input)
    readhead = ReadHead(
        max_bp=max_bp,
        hidden_dim=pca_hidden_dim,
        num_heads=pca_heads,
        num_layers=pca_layers,
        output_dim=1
    )
    # output = readhead(output)
    model = NucModel(
        max_bp,
        batch_size,
        nuc_emb,
        readhead,
        shift,
        scale
    )
    return model


@tf.keras.saving.register_keras_serializable(package="UnifracModel")
class UnifracModel(tf.keras.Model):
    def __init__(
        self,
        max_bp,
        batch_size,
        feature_emb,
        readhead,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_bp = max_bp
        self.batch_size = batch_size
        self.feature_emb = feature_emb
        self.readhead = readhead
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pair_loss = PairwiseLoss()
        self.metric_traker = PairwiseMAE()
        self.make_call_function()

    def make_call_function(self):
        @tf.autograph.experimental.do_not_convert
        def one_step(inputs, training=None):
            ind, seq = inputs
            output = self.feature_emb(seq, training=training)
            output = self.readhead(output)
            return output
        self.call_function = tf.function(one_step, reduce_retracing=True)

    def call(self, inputs, training=None):
        return self.call_function(inputs, training=training)

    def build(self, input_shape):
        input = tf.keras.Input(
            shape=[None, self.max_bp],
            batch_size=self.batch_size,
            dtype=tf.int64
        )
        output = self.feature_emb(input)
        output = self.readhead(output)

    def train_step(self, data):
        x, y = data
        ind, seq = tf.nest.flatten(x)
        ind = tf.squeeze(ind)
        y = tf.gather(y, ind, axis=1, batch_dims=0)

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self((ind, seq), training=True)

            # Compute regression loss
            loss = self.pair_loss(y, outputs)
            self.loss_tracker.update_state(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.metric_traker.update_state(y, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.metric_traker.result(),
        }

    def test_step(self, data):
        x, y = data
        ind, seq = tf.nest.flatten(x)
        ind = tf.squeeze(ind)
        y = tf.gather(y, ind, axis=1, batch_dims=0)
        # Forward pass
        outputs = self((ind, seq), training=False)

        # Compute regression loss
        loss = self.pair_loss(y, outputs)
        self.loss_tracker.update_state(loss)

        # Compute our own metrics
        self.metric_traker.update_state(y, outputs)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.metric_traker.result(),
        }

    def get_config(self):
        base_config = super().get_config()
        config = {
            "max_bp": self.max_bp,
            "batch_size": self.batch_size,
            "feature_emb": tf.keras.saving.serialize_keras_object(
                self.feature_emb
            ),
            "readhead": tf.keras.saving.serialize_keras_object(
                self.readhead
            ),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["feature_emb"] = tf.keras.saving.deserialize_keras_object(
            config["feature_emb"]
        )
        config["readhead"] = tf.keras.saving.deserialize_keras_object(
            config["readhead"]
        )
        return cls(**config)


@tf.keras.saving.register_keras_serializable(package="NucModel")
class NucModel(tf.keras.Model):
    def __init__(
        self,
        max_bp,
        batch_size,
        feature_emb,
        readhead,
        shift,
        scale,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_bp = max_bp
        self.batch_size = batch_size
        self.feature_emb = feature_emb
        self.readhead = readhead
        self.shift = shift
        self.scale = scale
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pair_loss = tf.keras.losses.MeanSquaredError()
        self.attention_loss = (
            tf.keras.losses.SparseCategoricalCrossentropy(
                ignore_class=0,
                from_logits=True,
                reduction="none"
            )
        )
        self.confidence_tracker = tf.keras.metrics.Mean(name="confidence")
        self.metric_traker = MAE(shift, scale, name="mae")
        self.make_call_function()

    def make_call_function(self):
        @tf.autograph.experimental.do_not_convert
        def one_step(inputs, training=None):
            ind, seq, rclr = inputs
            output, seq = self.feature_emb((seq, rclr), training=training)
            output = self.readhead(output)
            return (output, seq)
        self.call_function = tf.function(one_step, reduce_retracing=True)

    def call(self, inputs, training=None):
        return self.call_function(inputs, training=training)

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

    def predict_step(self, data):
        x, y = data
        ind, seq, rclr = tf.nest.flatten(x)
        ind = tf.squeeze(ind)
        outputs, seq = self((ind, seq, rclr), training=True)
        reg_out, logits = tf.nest.flatten(outputs, expand_composites=True)

        return reg_out

    def train_step(self, data):
        x, y = data
        ind, seq, rclr = tf.nest.flatten(x)
        ind = tf.squeeze(ind)

        with tf.GradientTape() as tape:
            # Forward pass
            outputs, seq = self((ind, seq, rclr), training=True)
            reg_out, logits = tf.nest.flatten(outputs, expand_composites=True)
            # Compute regression loss
            loss = self.pair_loss(y, reg_out)
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
        self.confidence_tracker.update_state(attention_loss)
        return {
            "loss": self.loss_tracker.result(),
            "confidence": self.confidence_tracker.result(),
            "mae": self.metric_traker.result(),
        }

    def test_step(self, data):
        x, y = data
        ind, seq, rclr = tf.nest.flatten(x)
        ind = tf.squeeze(ind)

        # Forward pass
        outputs, seq = self((ind, seq, rclr), training=False)
        reg_out, logits = tf.nest.flatten(outputs, expand_composites=True)

        # Compute regression loss
        loss = self.pair_loss(y, reg_out)
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
        self.confidence_tracker.update_state(attention_loss)
        return {
            "loss": self.loss_tracker.result(),
            "confidence": self.confidence_tracker.result(),
            "mae": self.metric_traker.result(),
        }

    def get_config(self):
        base_config = super().get_config()
        config = {
            "max_bp": self.max_bp,
            "batch_size": self.batch_size,
            "feature_emb": tf.keras.saving.serialize_keras_object(
                self.feature_emb
            ),
            "readhead": tf.keras.saving.serialize_keras_object(
                self.readhead
            ),
            "shift": self.shift,
            "scale": self.scale,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["feature_emb"] = tf.keras.saving.deserialize_keras_object(
            config["feature_emb"]
        )
        config["readhead"] = tf.keras.saving.deserialize_keras_object(
            config["readhead"]
        )
        return cls(**config)
