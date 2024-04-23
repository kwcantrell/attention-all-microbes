"""gotu.py"""
import json

import biom
import numpy as np
import tensorflow as tf

from aam.callbacks import SaveModel
from gotu.gotu_data_utils import create_training_dataset, create_prediction_dataset
from gotu.gotu_model import gotu_classification, gotu_predict


def run_gotu_training(asv_fp: str, gotu_fp: str, model_fp: None) -> None:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Emotional Support GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    training_dataset, validation_dataset, gotu_dict = create_training_dataset(gotu_fp,
                                                                  asv_fp)

    if model_fp is None:
        model = gotu_classification(batch_size=8,
                                    load_model=False,
                                    model_fp=model_fp,
                                    dropout=0.1,
                                    dff=256,
                                    d_model=128,
                                    enc_layers=4,
                                    enc_heads=8,
                                    max_bp=150)
    model.summary()
    history = model.fit(training_dataset, 
                        validation_data=validation_dataset, 
                        callbacks=[SaveModel("gotu_decoder_model")], epochs=1000)


def run_gotu_predictions(asv_fp: str, gotu_fp: str, model_fp: str, pred_out_path: str) -> None:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Emotional Support GPUs")
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
        
    combined_dataset, gotu_dict = create_prediction_dataset(gotu_path, asv_path, 8)
    model = gotu_predict(8, 
                        model_fp,
                        dropout=0,
                        dff=512,
                        d_model=128,
                        enc_layers=4,
                        enc_heads=8,
                        max_bp=150)
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
    with biom.util.biom_open(f"{pred_out_path}", 'w') as f:
        pred_biom.to_hdf5(f, "Predicted GOTUs Using DeepLearning")

asv_path = "../data/asv_ordered_table.biom"
gotu_path = "../data/gotu_ordered_table.biom"
model_fp = "../attention-all-microbes/gotu_decoder_model/20230321_dynamic_learning_encoder.keras"
pred_out_path = "../attention-all-microbes/gotu_decoder_model/tester.biom"

run_gotu_predictions(asv_fp=asv_path,
                     gotu_fp=gotu_path,
                     model_fp=model_fp,
                     pred_out_path=pred_out_path)