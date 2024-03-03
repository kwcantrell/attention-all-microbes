import tensorflow as tf
import tensorflow_models as tfm
from amplicon_gpt.losses import unifrac_loss, _pairwise_distances
from amplicon_gpt.layers import ReadHead, MultiHeadPCAProjection

MAX_SEQ = 1600
BATCH_SIZE = 8


def _construct_base(batch_size: int, output_dim: int):
    d_model = 64
    dff = 2048
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
    model_input = MultiHeadPCAProjection()(model_input)
    model_input = tfm.nlp.models.TransformerEncoder(
            num_layers=num_enc_layers,
            num_attention_heads=num_heads,
            intermediate_size=dff,
            dropout_rate=dropout,
            norm_first=True,
            activation='relu',
        )(model_input)
    output = ReadHead(d_model, output_dim=output_dim)(model_input)
    return tf.keras.Model(inputs=input, outputs=output)


def transfer_learn_base(batch_size: int):
    return _construct_base(batch_size, 64)


def classification(batch_size: int, dropout: float):
    pass


def regression(batch_size: int):
    model = _construct_base(batch_size, 1)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    model.compile(optimizer=optimizer,
                  loss='mse', metrics=['mae'],
                  jit_compile=False)
    return model


@tf.keras.saving.register_keras_serializable(package="amplicon_gpt.metrics")
class MAE(tf.keras.metrics.Metric):
    def __init__(self, name='mae_loss', dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        self.loss = self.add_weight(name='rl',
                                    initializer='zero',
                                    dtype=tf.float32)
        self.i = self.add_weight(name='i',
                                 initializer='zero',
                                 dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        pairwise_mae = tf.abs(_pairwise_distances(y_pred) - y_true)
        self.loss.assign_add(tf.reduce_sum(pairwise_mae))
        COMPARISONS = (BATCH_SIZE * BATCH_SIZE) - BATCH_SIZE / 2.0
        self.i.assign_add(tf.constant(COMPARISONS, dtype=tf.float32))

    def result(self):
        return self.loss / self.i


def compile_model(model, batch_size):
    # lr = 0.0001
    boundaries = [4400*4]
    values = [0.0001, 0.0001]
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                                                    boundaries,
                                                    values)

    # model.load_weights('base-model/encoder.keras')
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    model.compile(optimizer=optimizer,
                  loss=unifrac_loss(batch_size), metrics=[MAE()],
                  jit_compile=False)
    return model
