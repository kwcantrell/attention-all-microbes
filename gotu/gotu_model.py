"""gotu_model.py"""
import numpy as np
import biom
import tensorflow as tf
import tensorflow_models as tfm

from aam.callbacks import SaveModel
from gotu.gotu_callback import CustomSchedule, CSVCallback
from attention_regression.model import _construct_model

NUM_GOTUS = 6838

# TODO Currently model is broken until new construct model is added
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
    asv_model = _construct_model(
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

def run_gotu_predictions(
    asv_fp: str, gotu_fp: str, model_fp: str, pred_out_path: str
) -> None:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus),
                "Physical GPUs,",
                len(logical_gpus),
                "Emotional Support GPUs",
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    asv_biom = biom.load_table(asv_fp)
    gotu_biom = biom.load_table(gotu_fp)
    asv_samples = asv_biom.ids(axis="sample")
    gotu_ids = gotu_biom.ids(axis="observation")

    pred_biom_data = np.zeros((len(gotu_ids), len(asv_samples)))
    gotu_obv_dict = {}
    for i in range(len(gotu_ids)):
        gotu_obv_dict[gotu_ids[i]] = i

    combined_dataset, gotu_dict = create_prediction_dataset(gotu_fp, asv_fp, 8)
    model = gotu_predict(
        8,
        model_fp,
        dropout=0,
        dff=512,
        d_model=128,
        enc_layers=4,
        enc_heads=8,
        max_bp=150,
    )
    model.summary()

    col_index = 0
    for x, y in combined_dataset:
        predictions = model.predict_on_batch(x)
        predicted_classes = np.argmax(predictions, axis=-1)

        y_pred = predicted_classes
        for i in range(len(y_pred)):
            for j in range(len(y_pred[i, :])):
                gotu_code = y_pred[i, :][j]
                if gotu_code > 3:
                    gotu_name = gotu_dict[int(gotu_code)]
                    gotu_pos = gotu_obv_dict[gotu_name]
                    pred_biom_data[gotu_pos, col_index] = 1
            col_index += 1

    pred_biom = biom.table.Table(pred_biom_data, gotu_ids, asv_samples)
    with biom.util.biom_open(f"{pred_out_path}", "w") as f:
        pred_biom.to_hdf5(f, "Predicted GOTUs Using DeepLearning")

def run_gotu_training(
    training_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    **kwargs,
) -> None:
    model_fp = kwargs.get("model_fp", None)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus),
                "Physical GPUs,",
                len(logical_gpus),
                "Emotional Support GPUs",
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if model_fp is None:
        model = gotu_classification(
            batch_size=8,
            dropout=0.1,
            dff=256,
            d_model=128,
            enc_layers=4,
            enc_heads=8,
            max_bp=150,
        )
    model.summary()
    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        callbacks=[SaveModel("gotu_decoder_model")],
        epochs=1000,
    )