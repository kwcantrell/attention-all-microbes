import numpy as np
import tensorflow as tf
import tensorflow_models as tfm
from gotu.gotu_data_utils import create_datasets
from gotu.gotu_model import gotu_classification
from aam.callbacks import SaveModel

            
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Emotional Support GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Data preparation
asv_path = "../data/asv_ordered_table.biom"
gotu_path = "../data/gotu_ordered_table.biom"
training_dataset, validation_dataset, gotu_dict = create_datasets(gotu_path,
                                                                  asv_path)


model = gotu_classification(batch_size=8,
                            load_model=False,
                            dropout=0.1,
                            dff=512,
                            d_model=128,
                            enc_layers=6,
                            enc_heads=8,
                            max_bp=150)
model.summary()
history = model.fit(training_dataset, validation_data=validation_dataset, callbacks=[SaveModel("gotu_decoder_model")], epochs=1000)


