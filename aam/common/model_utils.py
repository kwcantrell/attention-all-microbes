import tensorflow as tf
import tensorflow_models as tfm
from aam.common.losses import PairwiseLoss
from aam.layers import ReadHead, NucleotideEmbedding
from aam.common.metrics import PairwiseMAE
from aam.models.nuc_model import _construct_base, _construct_regressor


# def _construct_base(
#         batch_size: int,
#         dropout_rate: float,
#         pca_hidden_dim: int,
#         pca_heads: int,
#         pca_layers: int,
#         dff: int,
#         token_dim: int,
#         ff_clr,
#         attention_layers: int,
#         attention_heads: int,
#         output_dim: int,
#         max_bp: int
# ):
#     input = tf.keras.Input(
#         shape=[None, max_bp],
#         batch_size=batch_size,
#         dtype=tf.int64
#     )
#     output = NucleotideEmbedding(
#         pca_hidden_dim,
#         max_bp,
#         pca_hidden_dim,
#         pca_heads,
#         pca_layers,
#         attention_heads,
#         attention_layers,
#         dff,
#         dropout_rate
#     )(input)
#     output = ReadHead(
#         hidden_dim=pca_hidden_dim,
#         num_heads=pca_heads,
#         num_layers=pca_layers,
#         output_dim=output_dim
#     )(output)
#     return tf.keras.Model(inputs=input, outputs=output)


def pretrain_unifrac(batch_size: int, lr: float, *args):
    model = _construct_base(batch_size, *args)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    model.compile(
        optimizer=optimizer,
        # loss=PairwiseLoss(),
        # metrics=[PairwiseMAE()],
        jit_compile=False
    )
    return model


def regressor(batch_size: int, lr: float, *args):
    model = _construct_regressor(batch_size, *args)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    model.compile(
        optimizer=optimizer,
        # loss=PairwiseLoss(),
        # metrics=[PairwiseMAE()],
        jit_compile=False
    )
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
    model.compile(optimizer=optimizer,
                  loss=pairwise_residual_mse(batch_size),
                  metrics=['mae'],
                  jit_compile=False)
    return model


def classification(batch_size: int, lr: float, *args):
    model = _construct_base(batch_size, *args)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'],
                  jit_compile=False)
    return model


def regression(batch_size: int, lr: float, *args):
    model = _construct_base(batch_size, *args)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    loss = pairwise_residual_mse(batch_size=batch_size)
    model.compile(optimizer=optimizer,
                  loss='mse', metrics=['mae'],
                  jit_compile=False)
    return model