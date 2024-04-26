import tensorflow as tf
from attention_regression.layers import (
    FeatureEmbedding,
    FeatureLoadings,
    Regressor
)
from aam.metrics import MAE


# temp hacky way to initialize model creation
def _construct_model(
    ids,
    shift,
    scale,
    token_dim,
    p_feature_attention_method,
    features_to_add,
    dropout,
    ff_clr,
    ff_d_model,
    pca_heads,
    enc_layers,
    enc_heads,
    lr
):
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

    feature_emb = FeatureEmbedding(
        token_dim,
        ids,
        p_feature_attention_method,
        features_to_add,
        ff_clr,
        ff_d_model,
        dropout,
        name="FeatureEmbeddings",
        dtype=tf.float32
    )
    emb_outputs = feature_emb((feature_input, rclr_input))

    output_embeddings = emb_outputs[2]
    output_regression = emb_outputs[3]

    binary_loadings = FeatureLoadings(
        enc_layers=enc_layers,
        enc_heads=enc_heads,
        dff=128,
        dropout=dropout,
        output_dim=(
            3 if p_feature_attention_method == 'add_features'
            else len(ids) + 1
        ),
        name="FeatureLoadings"
    )
    output_embeddings = binary_loadings(output_embeddings)

    regressor = Regressor(
        ff_d_model,
        pca_heads,
        2,
        4,
        128,
        0.1
    )
    output_regression = regressor(output_regression)

    model = AttentionRegression(
        shift,
        scale,
        feature_emb,
        p_feature_attention_method,
        binary_loadings,
        regressor,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    model.compile(
        optimizer=optimizer
    )
    return model


@tf.keras.saving.register_keras_serializable(package="AttentionRegression")
class AttentionRegression(tf.keras.Model):
    def __init__(
            self,
            shift,
            scale,
            feature_emb,
            feature_attention_method,
            binary_loadings,
            regressor,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.shift = shift
        self.scale = scale
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = MAE(shift, scale, name="mae")
        self.confidence_tracker = tf.keras.metrics.Mean(name="confidence")
        self.loss_reg = tf.keras.losses.MeanSquaredError()
        self.attention_loss = (
            tf.keras.losses.SparseCategoricalCrossentropy(
                ignore_class=0,
                from_logits=True,
                reduction="none"
            )
        )
        self.feature_emb = feature_emb
        self.feature_attention_method = feature_attention_method
        self.binary_loadings = binary_loadings
        self.regressor = regressor
        self.TRUE_CLASS = tf.constant(2, dtype=tf.int64)
        self.ADDED_CLASS = tf.constant(1, dtype=tf.int64)

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
            "_model_out_keys": [
                "token_mask",
                "tokens",
                "embeddings",
                "regression"
            ],
            "class_labels": {
                "true":  2,
                "added": 1
            },
            "total_tokens": self.feature_emb.total_tokens
        }

    def _get_inputs(self, x):
        return (x['feature'], x['rclr'])

    def predict_step(self, data):
        x, y = data
        inputs = self._get_inputs(x)
        y = y['reg_out']
        y_pred = self((inputs), training=False)
        return y_pred

    @tf.function
    def select_equal_class(self, inputs):
        embeddings, labels, max_added = inputs
        added_embeddings = embeddings[-max_added:]
        added_labels = labels[-max_added:]

        true_embeddings = embeddings[:max_added]
        true_labels = labels[:max_added]
        return (
            tf.concat((true_embeddings, added_embeddings), axis=0),
            tf.concat((true_labels, added_labels), axis=0),
        )

    def _add_feature_loss(self, model_outputs):
        # extract feature labels
        _, token_mask, tokens, embeddings = model_outputs
        labels = tf.add(
            tf.cast(tf.greater(tokens, 0), dtype=tf.int64),
            tf.cast(token_mask, dtype=tf.int64)
        )
        max_added = tf.repeat(
            tf.reduce_max(
                tf.reduce_sum(
                    tf.cast(
                        tf.equal(labels, 1),
                        dtype=tf.int64
                    ),
                    axis=1
                )
            ),
            repeats=[tf.shape(tokens)[0]],
            axis=0
        )

        embeddings, labels = tf.vectorized_map(
            self.select_equal_class,
            [embeddings, labels, max_added]
        )

        attention_loss = tf.reduce_mean(
            self.attention_loss(labels, embeddings),
            axis=-1
        )
        return attention_loss

    def _add_mask_loss(self, model_outputs):
        features, _, sample_tokens, embeddings = model_outputs
        feature_tokens = self.feature_emb.feature_tokens(features)
        masked_tokens = tf.math.not_equal(
            feature_tokens,
            sample_tokens
        )
        labels = tf.multiply(
            feature_tokens,
            tf.cast(masked_tokens, dtype=tf.int64)
        )
        embedding_mask = tf.cast(
            tf.greater(
                labels,
                0
            ),
            dtype=tf.float32
        )
        embeddings = tf.multiply(
            embeddings,
            tf.expand_dims(embedding_mask, axis=-1)
        )
        attention_loss = tf.reduce_sum(
            self.attention_loss(labels, embeddings),
            axis=-1
        )
        return attention_loss

    def _attention_loss(self, model_outputs):
        if self.feature_attention_method == 'add_features':
            return self._add_feature_loss(model_outputs)
        elif self.feature_attention_method == 'mask_features':
            return self._add_mask_loss(model_outputs)
        else:
            raise Exception(
                f'Invalid attention method {self.feature_attention_method}'
            )

    def train_step(self, data):
        x, y = data
        inputs = self._get_inputs(x)
        y = y['reg_out']

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self((inputs), training=True)

            # Compute regression loss
            loss = self.loss_reg(y, outputs["regression"])
            model_outputs = (
                inputs[0],
                outputs["token_mask"],
                outputs["tokens"],
                outputs["embeddings"]
            )
            attention_loss = self._attention_loss(model_outputs)
            loss += attention_loss
            self.loss_tracker.update_state(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.mae_metric.update_state(y, outputs["regression"])
        self.confidence_tracker.update_state(attention_loss)
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

    def feature_confidences(self, dataset):
        conf_dict = {
            "conf_keys": ["count", "age", "confidence"],
            "classes": ["true", "added"]
        }

        keys = conf_dict["conf_keys"]
        for c in conf_dict["classes"]:
            conf_dict[c] = {}
            for k in keys:
                if k == "age":
                    conf_dict[c][k] = {
                        "true_age": tf.zeros(
                            self.feature_emb.total_tokens,
                            dtype=tf.float32
                        ),
                        "predicted_age": tf.zeros(
                            self.feature_emb.total_tokens,
                            dtype=tf.float32
                        )
                    }
                else:
                    conf_dict[c][k] = tf.zeros(
                        self.feature_emb.total_tokens,
                        dtype=tf.float32
                    )

        def _normalize_age(tensor):
            return self.scale * tensor + self.shift

        def _process_class_masks(sample_token_mask, sample_tokens, cls_val):
            class_mask = tf.greater(sample_tokens, 0)
            classes = tf.cast(sample_token_mask, dtype=tf.int64) + 1
            classes *= tf.cast(class_mask, dtype=tf.int64)
            cls_mask = tf.equal(classes, cls_val)
            class_tokens = (
                sample_tokens * tf.cast(cls_mask, dtype=tf.int64)
            )
            return cls_mask, class_tokens

        def _map_sample_tokens(inputs):
            sampe_confidence, sample_tokens = inputs
            return tf.scatter_nd(
                tf.expand_dims(sample_tokens, axis=-1),
                sampe_confidence,
                shape=[self.feature_emb.total_tokens]
            )

        def map_to_tokens(
                batch_tensor,
                class_mask,
                class_tokens
        ):
            # extract class info from batch tensor
            class_tensor = tf.multiply(
                batch_tensor,
                tf.cast(class_mask, dtype=tf.float32)
            )

            # map the class_tensor to the respective token indices
            batch_token_tensor = tf.reduce_sum(
                tf.map_fn(
                    _map_sample_tokens,
                    (class_tensor, class_tokens),
                    fn_output_signature=tf.float32
                ),
                axis=0
            )
            return batch_token_tensor

        def _process_batch(x, y):
            inputs = self._get_inputs(x)
            model_outputs = self(inputs)

            model_outputs["predicted_age"] = _normalize_age(
                model_outputs["regression"]
            )
            model_outputs["true_age"] = _normalize_age(
                tf.expand_dims(
                    y["reg_out"],
                    axis=-1
                )
            )
            model_outputs["embeddings"] = tf.keras.activations.softmax(
                model_outputs["embeddings"]
            )

            for c in conf_dict["classes"]:
                class_mask, class_tokens = _process_class_masks(
                    model_outputs["token_mask"],
                    model_outputs["tokens"],
                    model_outputs["class_labels"][c]
                )

                cls_val = model_outputs["class_labels"][c]
                batch_confidence = tf.multiply(
                        model_outputs["embeddings"][:, :, cls_val],
                        tf.cast(class_mask, dtype=tf.float32)
                )
                conf_dict[c]["confidence"] += map_to_tokens(
                    batch_confidence,
                    class_mask,
                    class_tokens
                )

                ones = tf.ones_like(model_outputs["tokens"], dtype=tf.float32)

                conf_dict[c]["age"]["true_age"] += map_to_tokens(
                    tf.multiply(
                        model_outputs["true_age"],
                        ones
                    ),
                    class_mask,
                    class_tokens
                )
                conf_dict[c]["age"]["predicted_age"] += map_to_tokens(
                    tf.multiply(
                        model_outputs["predicted_age"],
                        ones
                    ),
                    class_mask,
                    class_tokens
                )

                conf_dict[c]["count"] += map_to_tokens(
                    ones,
                    class_mask,
                    class_tokens
                )

        for batch in dataset:
            _process_batch(*batch)

        for c in conf_dict["classes"]:
            class_dict = conf_dict[c]
            for k in conf_dict["conf_keys"]:
                if k == "count":
                    continue
                token_counts = class_dict["count"]
                if k == "age":
                    class_dict[k]["true_age"] /= token_counts
                    class_dict[k]["predicted_age"] /= token_counts
                else:
                    class_dict[k] /= token_counts
        return conf_dict

    def get_config(self):
        base_config = super().get_config()
        config = {
            "shift": self.shift,
            "scale": self.scale,
            "feature_emb": tf.keras.saving.serialize_keras_object(
                self.feature_emb
            ),
            "feature_attention_method": self.feature_attention_method,
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
