"""gotu_model.py"""

import biom
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm

from aam.callbacks import SaveModel
from aam.layers import InputLayer
from aam.nuc_model import BaseNucleotideModel
from aam.utils import LRDecrease


class TransferLearnNucleotideModel(BaseNucleotideModel):
    def __init__(
        self,
        base_model,
        dropout_rate,
        batch_size,
        max_bp,
        num_gotus,
        pca_hidden_dim,
        pca_heads,
        pca_layers,
        count_ff_dim=32,
        num_layers=2,
        num_attention_heads=8,
        dff=32,
        use_attention_loss=True,
        d_model=128,
        **kwargs,
    ):
        use_attention_loss = False
        super().__init__(
            batch_size=batch_size,
            use_attention_loss=use_attention_loss,
            **kwargs,
        )

        self.max_bp = max_bp
        self.dropout_rate = dropout_rate
        self.count_ff_dim = count_ff_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.dff = dff
        self.regresssion_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric_traker = tf.keras.metrics.Mean(name="loss")

        self.input_layer = InputLayer(name="input_layer")

        self.base_model = base_model
        self.base_model.trainable = False  # Freeze the base model

        self.decoder_embedding = tf.keras.layers.Embedding(
            input_dim=num_gotus + 4, output_dim=128, embeddings_initializer="uniform", input_length=1, name="decoder_embedding"
        )

        self.transformer_decoder = tfm.nlp.models.TransformerDecoder(
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=dff,
            norm_first=True,
            activation="relu",
        )

        self.dense_output = tf.keras.layers.Dense(num_gotus + 4)

    def call(self, inputs, training=False):
        encoder_inputs, decoder_inputs = self.input_layer(inputs)
        encoder_output = self.base_model(encoder_inputs, training=training)

        decoder_embeddings = self.decoder_embedding(decoder_inputs)
        decoder_embeddings = tf.squeeze(decoder_embeddings, axis=-2)

        transformer_output = self.transformer_decoder(decoder_embeddings, encoder_output, training=training)

        output = self.dense_output(transformer_output)
        return output

    def model_step(self, inputs, training=False):
        encoder_inputs, decoder_inputs = self.input_layer(inputs)
        output = self((encoder_inputs, decoder_inputs), training=training)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "base_model": tf.keras.saving.serialize_keras_object(self.base_model),
                "batch_size": self.batch_size,
                "dropout": self.dropout,
                "dff": self.dff,
                "d_model": self.d_model,
                "enc_layers": self.enc_layers,
                "enc_heads": self.enc_heads,
                "max_bp": self.max_bp,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        base_model = tf.keras.saving.deserialize_keras_object(config.pop("base_model"))
        return cls(base_model=base_model, **config)


def gotu_classification(batch_size: int, load_model: False, model_fp: None, **kwargs):
    model = gotu_model_base(batch_size, **kwargs)
    if load_model:
        model.load_weights(model_fp)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LRDecrease(),
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


def run_gotu_predictions(asv_fp: str, gotu_fp: str, model_fp: str, pred_out_path: str) -> None:
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
