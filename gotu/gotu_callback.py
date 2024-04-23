"""gotu_callback.py"""

import os

import pandas as pd
import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=400):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model}


class CSVCallback(tf.keras.callbacks.Callback):
    """CSV output of current validation"""

    def __init__(self, dataset, gotu_map, out_dir, report_back_after_epochs=5):
        super().__init__()
        self.dataset = dataset
        self.gotu_map = gotu_map
        self.out_dir = out_dir
        self.report_back_after_epochs = report_back_after_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.report_back_after_epochs == 0:
            for x, y in self.dataset:
                encoder_inputs = x["encoder_inputs"]
                decoder_inputs = x["decoder_inputs"]
                true_gotus = y
                break

            pred_val = tf.math.argmax(self.model(x), axis=-1)
            # true_val = np.array([ys for (_,ys) in self.dataset])

            pred_df = pd.DataFrame(data=pred_val, columns=["predicted_gotus"])
            pred_df.to_csv(os.path.join(self.out_dir, "we_tried.csv"))
