import tensorflow as tf
import tensorflow_models as tfm

from aam.layers.encoders.nucleotides import ReadHead
from aam.models.nuc_model import NucModel


def regressor(batch_size: int, lr: float, *args):
    model = NucModel(batch_size, *args)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    model.compile(
        optimizer=optimizer,
        jit_compile=False,
    )
    return model


# def pretrain_unifrac(batch_size: int, lr: float, *args):
#     model = _construct_base(batch_size, *args)

#     optimizer = tf.keras.optimizers.Adam(
#         learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9
#     )
#     model.compile(
#         optimizer=optimizer,
#         # loss=PairwiseLoss(),
#         # metrics=[PairwiseMAE()],
#         jit_compile=False,
#     )
#     return model


# def transfer_regression(batch_size: int, lr: float, *args):
#     model = _construct_base(batch_size, *args)
#     model.build(input_shape=[8, None, 100])
#     model.load_weights("base-model-large/encoder.keras")
#     model.trainable = False
#     optimizer = tf.keras.optimizers.Adam(
#         learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9
#     )
#     output = tfm.nlp.models.TransformerEncoder(
#         num_layers=4,
#         num_attention_heads=8,
#         intermediate_size=1024,
#         dropout_rate=0.1,
#         norm_first=True,
#         activation="relu",
#     )(model.layers[-2].output)
#     output = ReadHead(hidden_dim=128, num_heads=8, num_layers=1, output_dim=1)(output)
#     model = tf.keras.Model(inputs=model.layers[0].input, outputs=output)
#     loss = pairwise_residual_mse(batch_size=batch_size)
#     model.compile(
#         optimizer=optimizer,
#         loss=pairwise_residual_mse(batch_size),
#         metrics=["mae"],
#         jit_compile=False,
#     )
#     return model


# def classification(batch_size: int, lr: float, *args):
#     model = _construct_base(batch_size, *args)
#     optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, beta_2=0.999, epsilon=1e-7)
#     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     model.compile(
#         optimizer=optimizer, loss=loss, metrics=["accuracy"], jit_compile=False
#     )
#     return model


# def regression(batch_size: int, lr: float, *args):
#     model = _construct_base(batch_size, *args)
#     optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, beta_2=0.999, epsilon=1e-7)
#     loss = pairwise_residual_mse(batch_size=batch_size)
#     model.compile(optimizer=optimizer, loss="mse", metrics=["mae"], jit_compile=False)
#     return model
