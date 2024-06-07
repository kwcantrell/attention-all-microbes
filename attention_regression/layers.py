import tensorflow as tf
import tensorflow_models as tfm


def random_sequences_mask(seq, seeds, seq_mask_rate):
    random_mask = tf.random.stateless_uniform(tf.shape(seq), seeds[:, 0])
    random_mask = tf.cast(
        tf.math.greater_equal(random_mask, seq_mask_rate), dtype=tf.float32
    )
    seq = tf.multiply(seq, random_mask)
    return seq


def robust_clr(tensor):
    rclr_mask = float_mask(tensor)
    rclr_s_counts = tf.reduce_sum(rclr_mask, axis=-1, keepdims=True)
    gx = tf.exp(
        tf.math.divide_no_nan(
            tf.reduce_sum(
                tf.math.log(tensor + (1.0 - rclr_mask)), axis=-1, keepdims=True
            ),
            rclr_s_counts,
        )
    )
    tensor = tf.math.divide_no_nan(tensor, gx)
    rclr_mask = tf.cast(tf.equal(tensor, 0.0), dtype=tf.float32)
    tensor = tf.math.log(tensor + rclr_mask)
    return tensor


def float_mask(tensor):
    return tf.cast(tf.math.not_equal(tensor, 0), dtype=tf.float32)


def similarity_loss(features_per_selector):
    def loss_fn(tensor):
        similarity_mask = tf.cast(tf.greater(tensor, 0.0), dtype=tf.float32)
        loss = tf.reduce_sum(tf.multiply(tensor, similarity_mask), axis=-1)
        similarity_mask = tf.cast(
            tf.less(loss, features_per_selector), dtype=tf.float32
        )
        loss = tf.reduce_sum(tf.multiply(loss, similarity_mask))
        return loss

    return tf.function(loss_fn, jit_compile=True)


def feature_selector_loss(similarity_threshold=0.8):
    def loss_fn(tensor):
        similarity_mask = tf.cast(
            (tf.greater(tensor, similarity_threshold)), dtype=tf.float32
        )
        loss = tf.math.reduce_sum(tf.multiply(tensor, similarity_mask), axis=0)
        feature_matrix_loss_mask = tf.cast(tf.greater(loss, 1.0), dtype=tf.float32)
        loss = tf.reduce_sum(tf.multiply(loss, feature_matrix_loss_mask))
        return loss

    return tf.function(loss_fn, jit_compile=True)


def prevalence_loss():
    """Computes how prevalent the top_k features are"""

    def loss_fn(counts, top_k):
        total_counts = tf.reduce_sum(counts)
        filtered_counts = tf.gather(
            counts,
            top_k,
            axis=0,
        )
        filtered_counts = tf.reduce_sum(filtered_counts)
        loss = tf.subtract(total_counts, filtered_counts)
        return loss / total_counts

    return tf.function(loss_fn, jit_compile=True)


@tf.keras.saving.register_keras_serializable(package="activity_regularization")
class ActivityRegularizationLayer(tf.keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super().__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_mean(tf.math.square(inputs)))
        return inputs


@tf.keras.saving.register_keras_serializable(package="feature_selector")
class FeatureSelector(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_features, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.vocab_size = vocab_size - 1  # subtract one to account for <MASK> token
        self.features_per_selector = self.vocab_size / max_features
        self.max_features = max_features
        self.embeddings: tf.Variable = self.add_weight(
            "embeddings",
            shape=[self.vocab_size, d_model],
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=tf.float32,
        )
        self.selectors: tf.Variable = self.add_weight(
            "feature_selector",
            shape=[max_features, d_model],
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=tf.float32,
        )
        self.random_generator = tf.random.Generator.from_non_deterministic_state()
        self.random_sequences_mask = tf.function(random_sequences_mask)
        self.similarity_threshold = 0.95
        self.similarity_loss = similarity_loss(
            self.features_per_selector * self.similarity_threshold
        )
        self.feature_selector_loss = feature_selector_loss(self.similarity_threshold)
        self.prevalence_loss = prevalence_loss()

    def call(self, inputs, training=None):
        rclr = inputs
        input_shape = tf.shape(rclr)

        # normalize weights for cos-sim
        embeddings = tf.math.l2_normalize(self.embeddings, axis=-1)
        selectors = tf.math.l2_normalize(self.selectors, axis=-1)

        counts = tf.reduce_sum(float_mask(rclr), axis=0)
        count_mask = float_mask(counts)

        feature_matrix = tf.matmul(selectors, embeddings, transpose_b=True)
        filtered_feature_matrix = tf.multiply(
            feature_matrix, tf.expand_dims(count_mask, axis=0)
        )

        top_k = tf.math.argmax(filtered_feature_matrix, axis=-1, output_type=tf.int32)
        output_embeddings = tf.broadcast_to(
            selectors,
            shape=(input_shape[0], self.max_features, self.d_model),
        )

        loss = self.similarity_loss(feature_matrix)
        loss += self.feature_selector_loss(feature_matrix)
        loss += 0.1 * self.prevalence_loss(counts, top_k)
        self.add_loss(loss)

        return (output_embeddings, top_k)


@tf.keras.saving.register_keras_serializable(package="FeatureEmbedding")
class FeatureEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        token_dim,
        d_model,
        attention_heads,
        attention_layers,
        dff,
        dropout_rate,
        num_features_to_select=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.token_dim = token_dim
        # d_model = 32
        self.d_model = d_model
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.num_features_to_select = num_features_to_select
        self.tokens = tf.range(self.vocab_size)
        self.feature_selector = FeatureSelector(
            vocab_size, d_model, max_features=num_features_to_select
        )
        self.pos_emb = tfm.nlp.layers.PositionEmbedding(
            max_length=num_features_to_select + 1
        )

        # Obj 1 is AD prediction
        self.objectives = self.add_weight(
            "objectives",
            shape=[1, d_model],
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=tf.float32,
        )
        self.rclr_ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    d_model,
                    kernel_regularizer=tf.keras.regularizers.L2(),
                    bias_regularizer=tf.keras.regularizers.L2(),
                    activation="relu",
                ),
                tf.keras.layers.Dense(
                    d_model,
                    kernel_regularizer=tf.keras.regularizers.L2(),
                    bias_regularizer=tf.keras.regularizers.L2(),
                    activation="relu",
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        self.rclr_reg = ActivityRegularizationLayer(0.1)
        self.attention_layer = tfm.nlp.models.TransformerEncoder(
            num_layers=6,
            num_attention_heads=8,
            intermediate_size=512,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            intermediate_dropout=0.1,
            norm_first=True,
            activation="relu",
        )
        self.attention_reg = ActivityRegularizationLayer(0.1)
        self.robust_clr = tf.function(robust_clr)

    def call(self, inputs, training=False, mask=None, batch_size=32):
        indices, seq, rclr = tf.nest.flatten(inputs, expand_composites=True)
        seq = tf.scatter_nd(indices, seq, shape=(batch_size, self.vocab_size))

        # get top k feature set
        features, top_k = self.feature_selector(rclr, training=training)
        top_k = tf.repeat(
            tf.expand_dims(tf.squeeze(top_k), axis=0), repeats=[batch_size], axis=0
        )
        rclr = tf.gather(rclr, top_k, axis=1, batch_dims=1)
        rclr = tf.pad(rclr, [[0, 0], [0, 1]], constant_values=0)
        rclr = robust_clr(rclr)

        objectives = tf.broadcast_to(
            self.objectives, shape=(batch_size, 1, self.d_model)
        )
        features = tf.concat([features, objectives], axis=1)

        rclr = tf.expand_dims(rclr, axis=-1)
        rclr = self.rclr_reg(rclr)
        output = features + self.rclr_ff(rclr)
        output = output + self.pos_emb(output)
        output = self.attention_layer(
            output,
            training=training,
        )
        return self.attention_reg(output)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "vocab_size": self.vocab_size,
            "token_dim": self.token_dim,
            "d_model": self.d_model,
            "attention_heads": self.attention_heads,
            "attention_layers": self.attention_layers,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(package="ReadHead")
class ReadHead(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.reg_out = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    32,
                    kernel_regularizer=tf.keras.regularizers.L2(),
                    bias_regularizer=tf.keras.regularizers.L2(),
                    activation="relu",
                    use_bias=True,
                ),
                tf.keras.layers.Dense(output_dim),
            ]
        )
        self.act_reg = ActivityRegularizationLayer()

    def get_config(self):
        base_config = super().get_config()
        config = {"output_dim": self.output_dim}
        return {**base_config, **config}

    def call(self, inputs):
        reg_out = self.reg_out(inputs[:, -3, :])
        return reg_out
