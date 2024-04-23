"""gotu_model.py"""

import tensorflow as tf
import tensorflow_models as tfm

from gotu.gotu_callback import CustomSchedule, CSVCallback
from aam.model_utils import _construct_base

NUM_GOTUS = 6838


def gotu_model_base(
    batch_size: int,
    dropout: float,
    dff: int,
    d_model: int,
    enc_layers: int,
    enc_heads: int,
    max_bp: int,
):
    """base gotu model utilizing previous asv model"""
    asv_model = _construct_base(
        batch_size=batch_size,
        dropout=dropout,
        pca_hidden_dim=64,
        pca_heads=4,
        pca_layers=2,
        dff=dff,
        d_model=d_model,
        enc_layers=enc_layers,
        enc_heads=enc_heads,
        output_dim=32,
        max_bp=max_bp,
    )

    encoder_inputs = tf.keras.Input(
        shape=(None, 150),
        batch_size=batch_size,
        dtype=tf.int64,
        name="encoder_inputs",
    )
    decoder_inputs = tf.keras.Input(
        shape=(None, 1),
        batch_size=batch_size,
        dtype=tf.int64,
        name="decoder_inputs",
    )
    model = tf.keras.Model(
        inputs=asv_model.inputs, outputs=asv_model.layers[-2].output
    )

    model_output = model(encoder_inputs)

    decoder_embedding = tf.keras.layers.Embedding(
        NUM_GOTUS + 4,
        128,
        embeddings_initializer="uniform",
        input_length=1,
        name="decoder_embedding",
    )(decoder_inputs)
    decoder_embedding = tf.squeeze(decoder_embedding, axis=-2)

    model_output = tfm.nlp.models.TransformerDecoder(
        num_layers=enc_layers,
        dropout_rate=dropout,
        num_attention_heads=enc_heads,
        intermediate_size=dff,
        norm_first=True,
        activation="relu",
    )(decoder_embedding, model_output)

    decoder_outputs = tf.keras.layers.Dense(NUM_GOTUS + 4)(model_output)
    gotu_model = tf.keras.Model(
        inputs=(encoder_inputs, decoder_inputs),
        outputs=decoder_outputs,
        name="asv_to_gotu_classifier",
    )

    return gotu_model


def gotu_classification(
    batch_size: int, load_model: False, model_fp: None, **kwargs
):
    model = gotu_model_base(batch_size, **kwargs)
    if load_model:
        model.load_weights(model_fp)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=CustomSchedule(128),
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )
    model.compile(
        optimizer=optimizer,
        loss=loss_object,
        metrics=["sparse_categorical_accuracy"],
    )

    return model


def gotu_predict(batch_size: int, model_fp: str, **kwargs):
    model = gotu_model_base(batch_size, **kwargs)
    model.load_weights(model_fp)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=CustomSchedule(128),
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )
    model.compile(
        optimizer=optimizer,
        loss=loss_object,
        metrics=["sparse_categorical_accuracy"],
    )
    model.trainable = False

    return model
