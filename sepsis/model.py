import tensorflow as tf
from sepsis.losses import FeaturePresent
from sepsis.layers import FeatureEmbedding, PCA, ProjectDown, BinaryLoadings
from aam.metrics import MAE


@tf.keras.saving.register_keras_serializable(package="AttentionRegression")
class AttentionRegression(tf.keras.Model):
    def __init__(
            self,
            mean,
            std,
            feature_emb,
            binary_loadings,
            regressor,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = MAE(mean, std, name="mae")
        self.confidence_tracker = tf.keras.metrics.Mean(name="confidence")
        self.loss_reg = tf.keras.losses.MeanSquaredError()
        self.loss_loadings = tf.keras.losses.SparseCategoricalCrossentropy(
            ignore_class=0,
            from_logits=True
        )
        self.loss_feat_pred = FeaturePresent()

        self.feature_emb = feature_emb
        self.binary_loadings = binary_loadings
        self.regressor = regressor

    def call(self, inputs, training=None):
        emb_outputs = self.feature_emb(inputs, training=training)
        output_token_mask = emb_outputs[0]
        output_tokens = emb_outputs[1]
        output_embeddings = emb_outputs[2]
        output_regression = emb_outputs[3]

        output_embeddings = self.binary_loadings(output_embeddings, training)
        output_regression = self.regressor(output_regression)

        return {
            "token_mask": output_token_mask,
            "tokens": output_tokens,
            "embeddings": output_embeddings,
            "regression": output_regression,
        }

    def _get_inputs(self, x):
        return (x['feature'], x['rclr'])

    def predict_step(self, data):
        x, y = data
        inputs = self._get_inputs(x)
        y = y['reg_out']
        y_pred = self((inputs), training=False)
        return y_pred

    def train_step(self, data):
        x, y = data
        inputs = self._get_inputs(x)
        y = y['reg_out']

        with tf.GradientTape() as tape:
            outputs = self((inputs), training=True)  # Forward pass
            # Compute our own loss
            loss = self.loss_reg(y, outputs["regression"])

            # extract feature labels
            token_mask = outputs["token_mask"]
            tokens = outputs["tokens"]
            label_mask = tf.greater(tokens, 0)
            labels = tf.cast(token_mask, dtype=tf.int64) + 1
            labels *= tf.cast(label_mask, dtype=tf.int64)
            embeddings = outputs["embeddings"]
            conf_loss = self.loss_loadings(labels, embeddings)
            loss += conf_loss
    
            self.loss_tracker.update_state(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.mae_metric.update_state(y, outputs["regression"])
        self.confidence_tracker.update_state(conf_loss)
        return {
            "loss": self.loss_tracker.result(),
            "confidence": self.confidence_tracker.result(),
            "mae": self.mae_metric.result(),
        }

    def build(self, input_shape):
        feature_input = tf.keras.Input(
            shape=[None],
            dtype=tf.string,
            name="feature"
        )
        rclr_input = tf.keras.Input(
            shape=[None],
            dtype=tf.float32,
            name="rclr"
        )
        emb_outputs = self.feature_emb((feature_input, rclr_input))
        output_embeddings = emb_outputs[2]
        output_regression = emb_outputs[3]

        output_embeddings = self.binary_loadings(output_embeddings)
        output_regression = self.regressor(output_regression)

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [
            self.loss_tracker,
            self.confidence_tracker,
            self.mae_metric
        ]

    def _feature_count_helper(self, tokens, mask, token_range):
        tokens = tf.reshape(tokens * mask, shape=(1, -1))
        tokens = tokens[tokens > 0]
        tokens = tf.reduce_sum(
            tf.cast(
                tf.equal(tokens, token_range),
                dtype=tf.float32
            ),
            axis=1,
        )
        return tokens

    def get_token_indices(self, tokens, mask):
        tokens = tokens * mask
        return tf.cast(tokens[tokens > 0], dtype=tf.int64)

    def _feature_conf_helper(self, confidence, tokens, mask):
        max_token = tf.reduce_max(self.feature_emb.tokens + 1)
        confidence = tf.reshape(confidence * mask, shape=(-1, 1))
        token_indices = tf.reshape(self.get_token_indices(tokens, mask), shape=(-1, 1))

        confidence = tf.squeeze(tf.gather(confidence, token_indices))
        conf_score = tf.scatter_nd(
            token_indices,
            updates=confidence,
            shape=[max_token]
        )
        return conf_score

    def _feature_helper(self, age, pred_age, confidence, tokens, mask):
        conf_accum = self._feature_conf_helper(
            confidence,
            tokens,
            mask
        )

        ages = age * mask
        age_accum = self._feature_conf_helper(
            ages,
            tokens,
            mask
        )

        pred_ages = pred_age * mask
        pred_age_accum = self._feature_conf_helper(
            pred_ages,
            tokens,
            mask
        )
        return conf_accum, age_accum, pred_age_accum

    def feature_confidences(self, dataset):
        true_count_accum = tf.zeros_like(self.feature_emb.tokens, dtype=tf.float32)
        added_count_accum = tf.zeros_like(self.feature_emb.tokens, dtype=tf.float32)

        true_conf_accum = tf.zeros_like(self.feature_emb.tokens, dtype=tf.float32)
        added_conf_accum = tf.zeros_like(self.feature_emb.tokens, dtype=tf.float32)
        
        pred_true_age_accum = tf.zeros_like(self.feature_emb.tokens, dtype=tf.float32)
        pred_added_age_accum = tf.zeros_like(self.feature_emb.tokens, dtype=tf.float32)

        true_age_accum = tf.zeros_like(self.feature_emb.tokens, dtype=tf.float32)
        added_age_accum = tf.zeros_like(self.feature_emb.tokens, dtype=tf.float32)

        token_range = tf.range(tf.reduce_max(self.feature_emb.tokens) + 1)
        token_range = tf.reshape(token_range, shape=(-1, 1))
        token_range = tf.cast(token_range, dtype=tf.float32)
        for x, y in dataset:
            inputs = self._get_inputs(x)
            output = self(inputs)

            pred_age = tf.reshape(output["regression"], shape=(-1, 1))
            pred_age = pred_age * self.std + self.mean

            ages = tf.reshape(y["reg_out"], shape=(-1, 1))
            ages = ages * self.std + self.mean

            confidence = tf.keras.activations.softmax(
                output["embeddings"]
            )

            token_indices = output['tokens']
            tokens = tf.cast(token_indices, dtype=tf.float32)

            true_mask = tf.cast(output['token_mask'], dtype=tf.float32)
            added_mask = tf.cast(~output['token_mask'], dtype=tf.float32)

            true_conf_score = confidence[:, :, 1]
            added_conf_score = confidence[:, :, 0]

            true_count_accum += self._feature_count_helper(
                tokens,
                true_mask,
                token_range
            )
            added_count_accum += self._feature_count_helper(
                tokens,
                added_mask,
                token_range
            )

            true_conf, true_age, pred = self._feature_helper(
                ages,
                pred_age,
                true_conf_score,
                tokens,
                true_mask
            )
            true_conf_accum += true_conf
            true_age_accum += true_age
            pred_true_age_accum += pred

            added_conf, added_age, pred = self._feature_helper(
                ages,
                pred_age,
                added_conf_score,
                tokens,
                added_mask
            )
            added_conf_accum += added_conf
            added_age_accum += added_age
            pred_added_age_accum += pred

        true_indices = true_count_accum > 0
        avg_true_conf = true_conf_accum[true_indices] / true_count_accum[true_indices]
        avg_true_age = true_age_accum[true_indices] / true_count_accum[true_indices]
        avg_pred_true_age = pred_true_age_accum[true_indices] / true_count_accum[true_indices]
        true_features = token_range[true_indices]

        added_indices = added_count_accum > 0
        avg_added_conf = added_conf_accum[added_indices] / added_count_accum[added_indices]
        avg_added_age = added_age_accum[added_indices] / added_count_accum[added_indices]
        avg_pred_add_age = pred_added_age_accum[added_indices] / added_count_accum[added_indices]
        added_features = token_range[added_indices]

        return {
            "avg_true_conf": avg_true_conf,
            "avg_added_conf": avg_added_conf,
            "true_count_accum": true_count_accum[true_indices],
            "added_count_accum": added_count_accum[added_indices],
            "avg_true_age": avg_true_age,
            "avg_added_age": avg_added_age,
            "true_features": true_features,
            "added_features": added_features,
            "pred_true_age_accum": avg_pred_true_age,
            "avg_pred_add_age": avg_pred_add_age
        }

    def mean_absolute_error(
            self,
            dataset,
            from_logits=True
    ):
        mae = tf.constant(0, dtype=tf.float32)
        counts = tf.constant(0, dtype=tf.int64)
        for x, y in dataset:
            inputs = self._get_inputs(x)
            outputs = self(inputs)
            y_pred = tf.squeeze(outputs["regression"])
            y_true = tf.squeeze(y["reg_out"])
            if from_logits:
                y_pred = (y_pred*self.std) + self.mean
                y_true = (y_true*self.std) + self.mean
            mae += tf.reduce_sum(tf.abs(y_true - y_true))
            counts += tf.shape(y_pred)[0]
        mae += tf.reduce_sum(mae)
        counts = tf.cast(counts, dtype=tf.float32)
        return mae / counts

    def get_config(self):
        base_config = super().get_config()
        config = {
            "mean": self.mean,
            "std": self.std,
            "feature_emb": tf.keras.saving.serialize_keras_object(
                self.feature_emb
            ),
            "binary_loadings": tf.keras.saving.serialize_keras_object(
                self.binary_loadings
            ),
            "regressor": tf.keras.saving.serialize_keras_object(
                self.regressor
            ),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["feature_emb"] = tf.keras.saving.deserialize_keras_object(
            config["feature_emb"]
        )
        config["binary_loadings"] = tf.keras.saving.deserialize_keras_object(
            config["binary_loadings"]
        )
        config["regressor"] = tf.keras.saving.deserialize_keras_object(
            config["regressor"]
        )
        return cls(**config)
