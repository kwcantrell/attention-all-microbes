import tensorflow as tf
import tensorflow_models as tfm
from aam.losses import pairwise_loss, pairwise_residual_mse
from aam.layers import ReadHead, PCAProjector, NucleotideEinsum
from aam.metrics import pairwise_mae


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
    input = tf.keras.Input(shape=[None, max_bp],
                           batch_size=batch_size,
                           dtype=tf.int64)
    model_input = tf.keras.layers.Embedding(
        5,
        d_model,
        embeddings_initializer="uniform",
        input_length=max_bp,
        input_shape=[batch_size, None, max_bp],
        name="embedding")(input)
    pos_emb_input = tfm.nlp.layers.PositionEmbedding(
                max_length=max_bp,
                seq_axis=2
    )(model_input)
    model_input = model_input + pos_emb_input
    model_input = PCAProjector(hidden_dim=pca_hidden_dim,
                               num_heads=pca_heads,
                               num_layers=pca_layers)(model_input)
    model_input = tfm.nlp.models.TransformerEncoder(
            num_layers=enc_layers,
            num_attention_heads=enc_heads,
            intermediate_size=dff,
            intermediate_dropout=dropout,
            norm_first=True,
            activation='relu',
        )(model_input)
    output = ReadHead(hidden_dim=pca_hidden_dim,
                      num_heads=pca_heads,
                      num_layers=pca_layers,
                      output_dim=output_dim)(model_input)
    return tf.keras.Model(inputs=input, outputs=output)


def pretrain_unifrac(batch_size: int, lr: float, *args):
    model = _construct_base(batch_size, *args)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr,
                                          weight_decay=0.1,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    model.compile(optimizer=optimizer,
                  loss=pairwise_loss(batch_size),
                  metrics=[pairwise_mae(batch_size)],
                  jit_compile=False)
    return model


def transfer_regression(batch_size: int, lr: float, *args):
    model = _construct_base(batch_size, *args)
    model.trainable = False
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    output = tfm.nlp.models.TransformerEncoder(
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=1024,
            dropout_rate=0.2,
            norm_first=True,
            activation='relu',
        )(model.layers[-2].output)
    output = ReadHead(hidden_dim=128,
                      num_heads=16,
                      num_layers=2,
                      output_dim=1)(output)
    model = tf.keras.Model(inputs=model.layers[0].input, outputs=output)
    loss = pairwise_residual_mse(batch_size=batch_size)
    model.compile(optimizer=optimizer,
                  loss=loss,
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
