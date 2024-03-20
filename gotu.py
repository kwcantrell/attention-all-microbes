import tensorflow as tf
import tensorflow_models as tfm
from gotu.gotu_data_utils import generate_gotu_dataset, batch_dataset
from gotu.gotu_model import gotu_classification
from aam.callbacks import SaveModel

            
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Data preparation
asv_path = "../data/asv_ordered_table.biom"
gotu_path = "../data/gotu_ordered_table.biom"
dataset, gotu_dict = generate_gotu_dataset(gotu_path, asv_path)

size = dataset.cardinality().numpy()
batch_size = 8
train_size = int(size * 0.7 / batch_size) * batch_size

training_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
training_dataset = batch_dataset(training_dataset, batch_size)
val_data = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
validation_dataset = batch_dataset(val_data, batch_size)

model = gotu_classification(batch_size=8,
                            load_model=True,
                            dropout=0.3,
                            dff=1024,
                            d_model=128,
                            enc_layers=6,
                            enc_heads=8,
                            max_bp=150)
model.summary()
history = model.fit(training_dataset, validation_data=validation_dataset, callbacks=[SaveModel("gotu_decoder_model")], epochs=1000)


