import tensorflow as tf

from aam.models.feedforward import FeedForward
from aam.models.transformers import TransformerEncoder


@tf.keras.utils.register_keras_serializable(package="SampleLearner")
class SampleLearner(tf.keras.Model):
    def __init__(
        self,
        rank_dim,
        base_model,
        num_hidden_layers=2,
        dropout_rate=0.0,
        output_bias=0.0,
        sample_only=False,
        **kwargs,
    ):
        super(SampleLearner, self).__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.rank_dim = rank_dim
        self.sample_only = sample_only

        self.custom_metrics = []
        self.membership_metrics = []

        self.output_bias = output_bias
        self.base_model = base_model
        self.state_ready = False
        self.sample_only = False

        self.state_ready = False
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.membership_loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True)
        self.rank_loss = tf.keras.losses.CategoricalFocalCrossentropy(label_smoothing=0.1)

        self.loss_tracker = tf.keras.metrics.Mean()
        # self.custom_metrics = [
        #     # tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
        #     # tf.keras.metrics.Precision(name="precision"),
        #     # tf.keras.metrics.Recall(name="recall"),
        #     # tf.keras.metrics.AUC(name="auc"),
        #     # tf.keras.metrics.AUC(name="prc", curve="PR"),
        #     tf.keras.metrics.MeanAbsoluteError(name="mae")
        # ]

        self.membership_metrics = [
            tf.keras.metrics.BinaryAccuracy(name="member_accuracy", threshold=0.5),
            tf.keras.metrics.Precision(name="member_precision"),
            tf.keras.metrics.Recall(name="member_recall"),
            tf.keras.metrics.AUC(name="member_auc"),
            tf.keras.metrics.AUC(name="member_prc", curve="PR"),
        ]
        self.rank_metric = tf.keras.metrics.Mean(name="mean_rank_error")

    def build(self, input_shape):
        sample_inputs = input_shape[0]
        emb_dim = sample_inputs[2]

        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, 256),
            dtype=tf.float32,
            trainable=True,
            initializer="uniform",
        )

        self.project_ff = tf.keras.layers.Dense(256, activation="gelu", name="project_ff")
        self.ff = tf.keras.Sequential(
            [
                FeedForward(),
                tf.keras.layers.Dense(
                    1,
                    bias_initializer=tf.keras.initializers.Constant(self.output_bias),
                    use_bias=True,
                ),
            ],
            name="cls_ff",
        )
        self.membership_ff = tf.keras.Sequential(
            [
                FeedForward(),
                tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True),
            ],
            name="membership_ff",
        )
        self.ranks_ff = tf.keras.Sequential(
            [
                FeedForward(),
                tf.keras.layers.Dense(self.rank_dim, activation="softmax", use_bias=True, name="ranks"),
            ],
            name="ranks_ff",
        )

        self.encoder = TransformerEncoder(
            num_layers=6,
            intermediate_size=512,
            fix_bias_shape=False,
            use_sparse_positions=False,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.dropout_rate,
        )
        super(SampleLearner, self).build(input_shape)

    def call(self, inputs, training=False):
        """
        inputs: [B, A, N], B: batch_dim, A: # ASV in sample, N: nuctides,
        string tensor
        """
        embeddings, attention_mask, counts = inputs
        input_shape = tf.shape(embeddings, out_type=tf.int32)
        batch_dim = input_shape[0]
        embeddings = self.project_ff(embeddings)

        # add class token
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_dim, axis=0)
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)
        padded_attention_mask = tf.pad(attention_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1.0)

        # encoder members
        encodings = self.encoder(embeddings, mask=padded_attention_mask, training=training)

        # compute membership prop
        members = encodings[:, 1:]

        mem_preds = tf.squeeze(self.membership_ff(members), axis=-1)

        ###############################################################################################
        # if self.state_ready:
        #     member_loss = self._compute_membership_loss(membership_labels, mem_preds, attention_mask)
        #     self.add_loss(member_loss)
        ################################################################################################

        ranks_pred = self.ranks_ff(members)
        ####################################################################################################
        # # need to argsort twice to get rank
        # rank_labels = tf.argsort(counts, direction="DESCENDING", stable=True)
        # rank_labels = tf.cast(tf.argsort(rank_labels, stable=True), dtype=tf.float32)

        # # break ties
        # rank_labels = tf.expand_dims(rank_labels, axis=-1)
        # tie_mask = tf.cast(tf.expand_dims(counts, axis=-1) == tf.expand_dims(counts, axis=1), dtype=tf.float32)

        # rank_labels = tf.floor(tf.reduce_sum(rank_labels * tie_mask, axis=1) / tf.reduce_sum(tie_mask, axis=1))
        # rank_labels = tf.cast(rank_labels, dtype=tf.int32)
        # valid_mask = tf.cast(tf.expand_dims(membership_labels, axis=-1), dtype=tf.float32)

        # if self.state_ready:
        #     rank_loss, rank_diff = self._compute_relative_rank_loss(
        #         rank_labels,
        #         ranks_pred,
        #         attention_mask * valid_mask,
        #     )
        #     self.add_loss(rank_loss)
        #########################################################################################################

        # add rank loss / metrics
        encoding = encodings[:, 0]
        cls_output = self.ff(encoding, training=training)

        return cls_output, mem_preds, ranks_pred

    def _compute_membership_loss(self, membership_label, membership_pred, mask):
        mask = tf.ensure_shape(mask, [None, None, 1])
        mask = mask > 0
        membership_pred._keras_mask = mask
        loss = self.membership_loss(membership_label, membership_pred)

        mask = tf.squeeze(mask, axis=-1)
        membership_label = membership_label[mask]
        membership_pred = membership_pred[mask]
        for metric in self.membership_metrics:
            metric.update_state(membership_label, membership_pred)
        return loss

    def _compute_relative_rank_loss(self, rank_label, rank_preds, mask):
        mask = tf.ensure_shape(mask, [None, None, 1])
        rank_one_hot = tf.one_hot(rank_label, self.rank_dim, on_value=1.0, off_value=0.0)
        rank_preds._keras_mask = mask
        rank_loss = self.rank_loss(rank_one_hot, rank_preds)

        predicted_rank = tf.cast(tf.argmax(rank_preds, axis=-1), dtype=tf.float32)
        rank_label = tf.cast(rank_label, dtype=tf.float32)
        rank_diff = tf.abs(rank_label - predicted_rank)
        mask = tf.squeeze(mask, axis=-1) > 0
        self.rank_metric.update_state(rank_diff[mask])
        return rank_loss, rank_diff

    def _create_rank_labels(self, counts):
        rank_labels = tf.argsort(counts, direction="DESCENDING", stable=True)
        rank_labels = tf.cast(tf.argsort(rank_labels, stable=True), dtype=tf.float32)

        # break ties
        rank_labels = tf.expand_dims(rank_labels, axis=-1)
        tie_mask = tf.cast(tf.expand_dims(counts, axis=-1) == tf.expand_dims(counts, axis=1), dtype=tf.float32)

        rank_labels = tf.floor(tf.reduce_sum(rank_labels * tie_mask, axis=1) / tf.reduce_sum(tie_mask, axis=1))
        rank_labels = tf.cast(rank_labels, dtype=tf.int32)
        return rank_labels

    def predict_step(self, data):
        x, y = data

        output = self(x, return_rank_diffs=True, training=False)

        return output, y

    def compute_metric(self, y, y_pred):
        y_true = y
        # y_true = tf.reshape(y_true, shape=[-1])
        # y_pred = tf.reshape(y_pred, shape=[-1])
        for metric in self.custom_metrics:
            metric.update_state(y_true, y_pred)

    def compute_loss(self, y, y_pred):
        y_true = y
        loss = self.loss_fn(y_true, y_pred)
        return tf.reduce_mean(loss)

    def get_inputs(self, data):
        x, y = data
        return x, y

    def train_step(self, data):
        x, y = self.get_inputs(data)

        embeddings, attention_mask, counts = x
        membership_labels = counts > 0

        rank_labels = self._create_rank_labels(counts)
        valid_mask = tf.cast(tf.expand_dims(membership_labels, axis=-1), dtype=tf.float32)
        with tf.GradientTape() as tape:
            cls_output, mem_preds, ranks_pred = self(x, training=True)
            mem_loss = self._compute_membership_loss(membership_labels, mem_preds, attention_mask)
            rank_loss, _ = self._compute_relative_rank_loss(rank_labels, ranks_pred, attention_mask * valid_mask)
            loss = mem_loss + rank_loss
            # loss = tf.reduce_sum(self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        output = {metric.name: metric.result() for metric in self.metrics}
        output.update({"loss": self.loss_tracker.result()})
        return output

    def test_step(self, data):
        x, y = self.get_inputs(data)

        embeddings, attention_mask, counts = x
        membership_labels = counts > 0
        rank_labels = self._create_rank_labels(counts)
        valid_mask = tf.cast(tf.expand_dims(membership_labels, axis=-1), dtype=tf.float32)

        cls_output, mem_preds, ranks_pred = self(x, training=True)
        mem_loss = self._compute_membership_loss(membership_labels, mem_preds, attention_mask)
        rank_loss, _ = self._compute_relative_rank_loss(rank_labels, ranks_pred, attention_mask * valid_mask)
        loss = mem_loss + rank_loss
        self.loss_tracker.update_state(loss)

        output = {metric.name: metric.result() for metric in self.metrics}
        output.update({"loss": self.loss_tracker.result()})
        return output

    # def predict_step(self, data):
    #     x, y = data
    #     output = self(x, training=False)
    #     predictions = self.output_activation(output)
    #     y = tf.cast(y, dtype=tf.float32)
    #     pred_labels = tf.cast(predictions >= 0.5, dtype=tf.float32)
    #     correct = tf.cast(pred_labels == y, dtype=tf.float32)
    #     correct = tf.reduce_mean(correct, axis=0)
    #     return predictions, y, correct

    def get_config(self):
        config = super(SampleLearner, self).get_config()
        config.update(
            {
                "build_input_shape": self.get_build_config(),
                "num_hidden_layers": self.num_hidden_layers,
                "dropout_rate": self.dropout_rate,
                "output_bias": self.output_bias,
                "rank_dim": self.rank_dim,
                "sample_only": self.sample_only,
                "base_model": self.base_model,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        build_input_shape = config.pop("build_input_shape")
        input_shape = build_input_shape["input_shape"]
        model = cls(**config)
        model.build(input_shape)
        return model
