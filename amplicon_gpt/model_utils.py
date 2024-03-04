import tensorflow as tf
import tensorflow_models as tfm
from amplicon_gpt.losses import pairwise_loss
from amplicon_gpt.layers import ReadHead, MultiHeadPCAProjection
from amplicon_gpt.metrics import pairwise_mae


def _construct_base(batch_size: int,
                    pca_heads,
                    t_heads,
                    output_dim: int):
    d_model = 64
    dff = 2048
    hidden_dim = 256
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
                                         num_heads=pca_heads,
                                         dropout=dropout)(model_input)
    model_input += tfm.nlp.layers.PositionEmbedding(
                max_length=5000,
                seq_axis=1)(model_input)
    model_input = tfm.nlp.models.TransformerEncoder(
            num_layers=num_enc_layers,
            num_attention_heads=t_heads,
            intermediate_size=dff,
            dropout_rate=dropout,
            norm_first=True,
            activation='relu',
        )(model_input)
    output = ReadHead(hidden_dim=dff,
                      num_heads=pca_heads,
                      output_dim=output_dim,
                      dropout=dropout)(model_input)
    return tf.keras.Model(inputs=input, outputs=output)


def pretrain_unifrac(batch_size: int):
    model = _construct_base(batch_size,
                            pca_heads=8,
                            t_heads=4,
                            output_dim=64)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    model.compile(optimizer=optimizer,
                  loss=pairwise_loss(batch_size),
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
    model = _construct_base(batch_size,
                            pca_heads=8,
                            t_heads=4,
                            output_dim=1)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001,
                                          beta_2=0.999,
                                          epsilon=1e-7)
    model.compile(optimizer=optimizer,
                  loss='mse', metrics=['mae'],
                  jit_compile=False)
    return model
