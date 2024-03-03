import tensorflow as tf
import tensorflow_models as tfm
from amplicon_gpt.losses import unifrac_loss, _pairwise_distances
from amplicon_gpt.layers import ReadHead, MultiHeadPCAProjection


def _construct_base(batch_size: int, output_dim: int):
    d_model = 64
    dff = 2048
    hidden_dim = 256
    num_heads = 4
    num_enc_layers = 6
    dropout = 0.5

    input = tf.keras.Input(shape=[None, 100],
                           batch_size=batch_size,
                           dtype=tf.int64)
    model_input = tf.keras.layers.Embedding(
        5,
        d_model,
        embeddings_initializer="uniform",
        input_length=100,
        input_shape=[batch_size, None, 100],
        name="embedding")(input)
    model_input = MultiHeadPCAProjection(hidden_dim=hidden_dim,
                                         dropout=dropout)(model_input)
    model_input += tfm.nlp.layers.PositionEmbedding(
                max_length=5000,
                seq_axis=1)(model_input)
    model_input = tfm.nlp.models.TransformerEncoder(
            num_layers=num_enc_layers,
            num_attention_heads=num_heads,
            intermediate_size=dff,
            dropout_rate=dropout,
            norm_first=True,
            activation='relu',
        )(model_input)
    output = ReadHead(hidden_dim=dff,
                      output_dim=output_dim,
                      dropout=dropout)(model_input)
    return tf.keras.Model(inputs=input, outputs=output)


def transfer_learn_base(batch_size: int):
    model = _construct_base(batch_size, 64)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    model.compile(optimizer=optimizer,
                  loss=unifrac_loss(batch_size),
                  metrics=[pairwise_mae(batch_size)],
                  jit_compile=False)
    return model


def classification(num_class: int, batch_size: int):
    model = _construct_base(batch_size, num_class)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[tf.keras.metrics.Accuracy()],
                  jit_compile=False)
    return model


def regression(batch_size: int):
    model = _construct_base(batch_size, 1)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    model.compile(optimizer=optimizer,
                  loss='mse', metrics=['mae'],
                  jit_compile=False)
    return model


def pairwise_mae(batch_size):
    @tf.keras.saving.register_keras_serializable(
        package="amplicon_gpt.metrics")
    class PairwiseMAE(tf.keras.metrics.Metric):
        def __init__(self, name='pairwise_mae', dtype=tf.float32):
            super().__init__(name=name, dtype=dtype)
            self.loss = self.add_weight(name='rl',
                                        initializer='zero',
                                        dtype=tf.float32)
            self.i = self.add_weight(name='i',
                                     initializer='zero',
                                     dtype=tf.float32)

        def update_state(self, y_true, y_pred):
            pairwise_mae = tf.abs(_pairwise_distances(y_pred) - y_true)
            self.loss.assign_add(tf.reduce_sum(pairwise_mae))
            COMPARISONS = (batch_size * batch_size) - batch_size / 2.0
            self.i.assign_add(tf.constant(COMPARISONS, dtype=tf.float32))

        def result(self):
            return self.loss / self.i

    return PairwiseMAE()
