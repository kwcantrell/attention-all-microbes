import tensorflow as tf
import tensorflow_models as tfm
from aam.losses import pairwise_loss, pairwise_residual_mse
from aam.layers import (
    ReadHead, PCAProjector, NucleotideEinsum, NucleotideEmbedding,
    PCA, ProjectDown
)
from aam.metrics import pairwise_mae


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=10000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model
        }


def _construct_base(batch_size: int,
                    dropout: float,
                    pca_hidden_dim: int,
                    pca_heads: int,
                    pca_layers: int,
                    dff: int,
                    d_model: int,
                    enc_layers: int,
                    enc_heads: int,
                    output_dim: int,
                    max_bp: int):
    input_asv = tf.keras.Input(
        shape=[None, max_bp],
        batch_size=batch_size,
        dtype=tf.int64,
        name='asvs'
    )
    input_clr = tf.keras.Input(
        shape=[None],
        batch_size=batch_size,
        dtype=tf.float32,
        name='clr'
    )
    model_input = NucleotideEmbedding(
        max_bp,
        512,
        d_model,
        pca_hidden_dim,
        pca_heads,
        pca_layers,
        dropout
    )((input_asv, input_clr))
    model_input = tfm.nlp.models.TransformerEncoder(
            num_layers=enc_layers,
            num_attention_heads=enc_heads,
            intermediate_size=dff,
            dropout_rate=dropout,
            norm_first=True,
            activation='relu',
    )(model_input)
    # model_input = tf.math.multiply(
    #     model_input,
    #     tf.expand_dims(input_clr, axis=-1)
    # )
    # model_input = tf.keras.layers.LayerNormalization()(model_input)
    output =  tf.keras.Sequential([
        ReadHead(
            hidden_dim=pca_hidden_dim,
            num_heads=pca_heads,
            num_layers=pca_layers,
            output_dim=output_dim,
            name='y_true'
        ),
    ])(model_input)
    return tf.keras.Model(inputs=(input_asv, input_clr), outputs=output)


def pretrain_unifrac(batch_size: int, lr: float, *args):
    model = _construct_base(batch_size, *args)
    
    # optimizer = tf.keras.optimizers.Adam(learning_rate=CustomSchedule(128),
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                          beta_1=0.9,
                                          beta_2=0.98,
                                          epsilon=1e-9)
    model.compile(optimizer=optimizer,
                  loss=pairwise_loss(batch_size),
                  metrics=[pairwise_mae(batch_size)],
                  jit_compile=False)
    return model


def transfer_regression(batch_size: int, lr: float, *args):
    model = _construct_base(batch_size, *args)
    model.build(input_shape=[8, None, 100])
    model.load_weights('base-model-large/encoder.keras')
    model.trainable = False
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                          beta_1=0.9,
                                          beta_2=0.98,
                                          epsilon=1e-9)
    output = tfm.nlp.models.TransformerEncoder(
            num_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            dropout_rate=0.1,
            norm_first=True,
            activation='relu',
        )(model.layers[-2].output)
    output = ReadHead(hidden_dim=128,
                      num_heads=8,
                      num_layers=1,
                      output_dim=1)(output)
    model = tf.keras.Model(inputs=model.layers[0].input, outputs=output)
    loss = pairwise_residual_mse(batch_size=batch_size)
    model.compile(
        optimizer=optimizer,
        loss=pairwise_residual_mse(batch_size),
        metrics=['mae'],
        jit_compile=False
    )
    return model


def classification(batch_size: int, lr: float, *args):
    model = _construct_base(batch_size, *args)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr,
        beta_2=0.999,
        epsilon=1e-7
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],
        jit_compile=False
    )
    return model


def regression(batch_size: int, lr: float, *args):
    model = _construct_base(batch_size, *args)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    loss = pairwise_residual_mse(batch_size=batch_size)
    model.compile(
        optimizer=optimizer,
        loss='mse', metrics=['mae'],
        jit_compile=False
    )
    return model
