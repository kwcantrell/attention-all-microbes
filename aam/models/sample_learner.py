import tensorflow as tf

from aam.models.feedforward import FeedForward
from aam.models.transformers import TransformerEncoder


class BatchToken(tf.keras.Model):
    def call(self, inputs):
        embeddings, attention_mask, cls_tkn = inputs
        input_shape = tf.shape(embeddings, out_type=tf.int32)
        batch_dim = input_shape[0]

        cls_tokens = tf.repeat(cls_tkn, repeats=batch_dim, axis=0)
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)
        padded_attention_mask = tf.pad(
            attention_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1.0
        )
        return embeddings, padded_attention_mask


@tf.keras.utils.register_keras_serializable(package="SampleLearner")
class SampleLearner(tf.keras.Model):
    def __init__(
        self,
        rank_dim,
        base_model,
        num_hidden_layers=2,
        dropout_rate=0.0,
        output_bias=0.0,
        output_std=1.0,
        sample_only=True,
        **kwargs,
    ):
        super(SampleLearner, self).__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.rank_dim = rank_dim
        self.sample_only = sample_only

        self.membership_metrics = []

        self.output_bias = output_bias
        self.output_std = output_std
        self.base_model = base_model
        self.state_ready = False
        self.state_ready = False
        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction="none")
        self.membership_loss = tf.keras.losses.BinaryFocalCrossentropy(
            reduction="none", apply_class_balancing=True
        )
        self.rank_loss = tf.keras.losses.CategoricalFocalCrossentropy(
            reduction="none"
        )
        # self.rank_loss = tf.keras.losses.CategoricalCrossentropy(
        #     reduction="none"
        # )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.sample_metrics = [
            # tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
            # tf.keras.metrics.Precision(name="precision"),
            # tf.keras.metrics.Recall(name="recall"),
            # tf.keras.metrics.AUC(name="auc"),
            # tf.keras.metrics.AUC(name="prc", curve="PR"),
        ]

        self.membership_metrics = [
            tf.keras.metrics.BinaryAccuracy(
                name="member_accuracy", threshold=0.5
            ),
            tf.keras.metrics.Precision(name="member_precision"),
            tf.keras.metrics.Recall(name="member_recall"),
            tf.keras.metrics.AUC(name="member_auc"),
            tf.keras.metrics.AUC(name="member_prc", curve="PR"),
        ]
        self.rank_metric = tf.keras.metrics.Mean(name="mean_rank_error")

        self.membership_ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="membership_ff",
        )
        self.ranks_ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.rank_dim, activation="softmax", name="ranks"
                ),
            ],
            name="ranks_ff",
        )
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, 256),
            dtype=tf.float32,
            trainable=True,
            initializer="uniform",
        )
        self.class_encoder = TransformerEncoder(
            num_layers=1,
            intermediate_size=512,
            fix_bias_shape=False,
            use_sparse_positions=False,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
        )

        self.encoder = TransformerEncoder(
            num_layers=4,
            intermediate_size=512,
            fix_bias_shape=False,
            use_sparse_positions=False,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.dropout_rate,
        )
        self.project_ff = tf.keras.layers.Dense(256, activation="gelu")
        self.ff = tf.keras.layers.Dense(1, use_bias=True, name="cls_ff")
        self.batch_token = BatchToken()

    def build_graph(self, input_shape):
        xs = [tf.keras.layers.Input(shape=shape[1:]) for shape in input_shape]
        return tf.keras.Model(inputs=[xs], outputs=self.call(xs))

    def build(self, input_shape):
        # ensure all layers are initialized
        self.build_graph(input_shape)
        super(SampleLearner, self).build(input_shape)
        self.built = True

    def compile(self, sample_only, output_bias=None, output_std=None, **kwargs):
        if output_bias is not None:
            self.output_bias = output_bias

        if output_std is not None:
            self.output_std = output_std

        self.sample_metrics = [
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
        ]
        self.sample_only = sample_only
        if not self.sample_only:
            self.project_ff.trainable = False
            self.encoder.trainable = False
            # self.membership_ff.trainable = False
            # self.ranks_ff.trainable = False

            # self.ff = tf.keras.Sequential(
            #     [
            #         tf.keras.layers.Dense(1, use_bias=True),
            #     ],
            #     name="cls_ff",
            # )

        super().compile(**kwargs)

    def call(self, inputs, training=False):
        """
        inputs: [B, A, N], B: batch_dim, A: # ASV in sample, N: nuctides,
        string tensor
        """
        embeddings, attention_mask, counts = inputs

        embeddings = self.project_ff(embeddings)
        embeddings = self.encoder(
            embeddings, mask=attention_mask, training=training
        )

        if not self.sample_only:
            print("Encoding those samples...")
            # embeddings, padded_attention_mask = self.batch_token(
            #     [embeddings, attention_mask, self.cls_token]
            # )
            embeddings = self.class_encoder(
                embeddings, mask=attention_mask, training=training
            )
            members = embeddings  # [:, 1:]
        else:
            members = embeddings

        mem_preds = tf.squeeze(
            self.membership_ff(members, training=training), axis=-1
        )
        ranks_pred = self.ranks_ff(members, training=training)

        # encoding = embeddings[:, 0]
        # encoding = self.ff_dropout(encoding, training=training)
        encoding = tf.reduce_sum(
            embeddings * attention_mask, axis=1
        ) / tf.reduce_sum(attention_mask, axis=1)
        # if not self.sample_only:
        #     encoding = self.norm(encoding)
        output = self.ff(encoding, training=training)
        return output, mem_preds, ranks_pred

    def _apply_loss_mask(self, loss, mask):
        mask = tf.squeeze(mask, axis=-1)
        loss = loss * mask
        loss = tf.reduce_sum(loss, axis=1) / tf.reduce_sum(mask, axis=1)
        return tf.reduce_mean(loss)

    def _compute_membership_loss(self, membership_label, membership_pred, mask):
        mask = tf.ensure_shape(mask, [None, None, 1])
        # mask = mask > 0
        # membership_pred._keras_mask = mask
        membership_label = tf.expand_dims(membership_label, axis=-1)
        membership_pred = tf.expand_dims(membership_pred, axis=-1)
        loss = self.membership_loss(membership_label, membership_pred)
        loss = self._apply_loss_mask(loss, mask)

        mask = mask > 0
        mask = tf.squeeze(mask, axis=-1)
        membership_label = membership_label[mask]
        membership_pred = membership_pred[mask]
        for metric in self.membership_metrics:
            metric.update_state(membership_label, membership_pred)
        return loss

    def _add_rank_noise(self, rank_one_hot):
        rank_one_hot = rank_one_hot * 0.9
        rank_indices = tf.expand_dims(tf.where(rank_one_hot > 0), axis=1)
        one_away = rank_indices + tf.constant(
            [[[0, 0, -1], [0, 0, 1]]], dtype=tf.int64
        )
        one_away = tf.reshape(one_away, shape=[-1, 3])
        one_away = tf.where(one_away > 0, one_away, 0)
        one_away = tf.where(
            one_away < self.rank_dim, one_away, self.rank_dim - 1
        )
        num_indices = tf.shape(one_away)[0]
        rank_one_hot = tf.tensor_scatter_nd_add(
            rank_one_hot,
            one_away,
            tf.ones([num_indices], dtype=tf.float32) * 0.025,
        )

        two_away = rank_indices + tf.constant(
            [[[0, 0, -2], [0, 0, 2]]], dtype=tf.int64
        )
        two_away = tf.reshape(two_away, shape=[-1, 3])
        two_away = tf.where(two_away > 0, two_away, 0)
        two_away = tf.where(
            two_away < self.rank_dim, two_away, self.rank_dim - 1
        )
        num_indices = tf.shape(two_away)[0]
        rank_one_hot = tf.tensor_scatter_nd_add(
            rank_one_hot,
            two_away,
            tf.ones([num_indices], dtype=tf.float32) * 0.0125,
        )
        return rank_one_hot

    def _compute_relative_rank_loss(
        self, rank_label, rank_preds, mask, mae=True
    ):
        mask = tf.ensure_shape(mask, [None, None, 1])
        rank_one_hot = tf.one_hot(
            rank_label, self.rank_dim, on_value=1.0, off_value=0.0
        )
        rank_one_hot = self._add_rank_noise(rank_one_hot)
        predicted_rank = tf.cast(
            tf.argmax(rank_preds, axis=-1), dtype=tf.float32
        )
        rank_loss = self.rank_loss(rank_one_hot, rank_preds)
        rank_label = tf.cast(rank_label, dtype=tf.float32)
        rank_diff = predicted_rank - rank_label

        # rank_weight = tf.minimum(
        #     tf.pow(0.99, -tf.cast(tf.abs(rank_diff), dtype=tf.float32)), 2.5
        # )
        # rank_loss = rank_loss * rank_weight
        rank_loss = self._apply_loss_mask(rank_loss, mask)

        if mae:
            rank_diff = tf.abs(rank_diff)
        mask = tf.squeeze(mask, axis=-1) > 0
        self.rank_metric.update_state(rank_diff[mask])
        mask = tf.cast(mask, dtype=tf.float32)
        return rank_loss, rank_diff, predicted_rank, mask

    def _create_rank_labels(self, counts):
        rank_labels = tf.argsort(counts, direction="DESCENDING", stable=True)
        rank_labels = tf.cast(
            tf.argsort(rank_labels, stable=True), dtype=tf.float32
        )

        # break ties
        rank_labels = tf.expand_dims(rank_labels, axis=-1)
        tie_mask = tf.cast(
            tf.expand_dims(counts, axis=-1) == tf.expand_dims(counts, axis=1),
            dtype=tf.float32,
        )

        tie_ranks = rank_labels * tie_mask
        # tie_ranks = tf.where(tie_mask > 0, tie_ranks, self.rank_dim)
        # rank_labels = tf.math.reduce_max(
        #     tie_ranks, axis=1
        # )  # - tf.math.floordiv(tf.reduce_sum(tie_mask), 2)
        # # # tf.print(rank_labels, counts)
        rank_labels = tf.math.ceil(
            tf.reduce_sum(tie_ranks, axis=1) / tf.reduce_sum(tie_mask, axis=1)
        )
        # rank_labels = tf.where(counts > 0, rank_labels, self.rank_dim - 1)
        rank_labels = tf.cast(rank_labels, dtype=tf.int32)
        return rank_labels

    def predict_step(self, data):
        x, y = data
        _, attention_mask, counts = x
        output, _, rank_pred = self(x, training=False)

        if self.sample_only:
            membership_labels = counts > 0
            rank_labels = self._create_rank_labels(counts)
            valid_mask = tf.cast(
                tf.expand_dims(membership_labels, axis=-1), dtype=tf.float32
            )

            _, rank_diff, predicted_ranks, mask = (
                self._compute_relative_rank_loss(
                    rank_labels,
                    rank_pred,
                    attention_mask,
                    mae=False,
                )
            )
            return (
                rank_diff,
                predicted_ranks,
                rank_labels,
                counts,
                mask,
                output,
                y,
            )
        else:
            return output, y

    def compute_metric(self, y, y_pred):
        y_true = y
        for metric in self.sample_metrics:
            metric.update_state(y_true, y_pred)

    def compute_loss(self, y, y_pred):
        y = tf.cast(y, dtype=tf.float32)
        loss = self.loss_fn(y, y_pred)
        # loss = tf.math.log(loss)
        return tf.math.log(tf.reduce_mean(loss))

    def get_inputs(self, data):
        x, y = data
        return x, y

    def train_step(self, data):
        x, y = self.get_inputs(data)

        embeddings, attention_mask, counts = x
        membership_labels = counts > 0

        rank_labels = self._create_rank_labels(counts)
        valid_mask = tf.cast(
            tf.expand_dims(membership_labels, axis=-1), dtype=tf.float32
        )
        with tf.GradientTape() as tape:
            output, mem_preds, ranks_pred = self(x, training=True)
            mem_loss = self._compute_membership_loss(
                membership_labels, mem_preds, attention_mask
            )
            rank_loss, _, _, _ = self._compute_relative_rank_loss(
                rank_labels, ranks_pred, attention_mask
            )
            loss = mem_loss + rank_loss
            if not self.sample_only:
                loss += self.compute_loss(y, output)
                self.compute_metric(y, output)
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
        valid_mask = tf.cast(
            tf.expand_dims(membership_labels, axis=-1), dtype=tf.float32
        )

        output, mem_preds, ranks_pred = self(x, training=False)
        mem_loss = self._compute_membership_loss(
            membership_labels, mem_preds, attention_mask
        )
        rank_loss, _, _, _ = self._compute_relative_rank_loss(
            rank_labels, ranks_pred, attention_mask
        )
        loss = mem_loss + rank_loss
        if not self.sample_only:
            loss += self.compute_loss(y, output)
            self.compute_metric(y, output)
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
                # "build_input_shape": self.get_build_config(),
                "num_hidden_layers": self.num_hidden_layers,
                "dropout_rate": self.dropout_rate,
                "output_bias": self.output_bias,
                "output_std": self.output_std,
                "rank_dim": self.rank_dim,
                "sample_only": self.sample_only,
                "base_model": self.base_model,
            }
        )
        return config

    # @classmethod
    # def from_config(cls, config):
    #     build_input_shape = config.pop("build_input_shape")
    #     input_shape = build_input_shape["input_shape"]
    #     model = cls(**config)
    #     model.build(input_shape)
    #     return model
