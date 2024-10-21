from __future__ import annotations

import datetime
import os

import numpy as np
import tensorflow as tf

from aam.callbacks import SaveModel
from aam.models.utils import cos_decay_with_warmup


class CVModel:
    def __init__(
        self, model: tf.keras.Model, train_data, val_data, output_dir, fold_label
    ):
        self.model: tf.keras.Model = model
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = output_dir
        self.fold_label = fold_label
        self.time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(
            output_dir,
            f"logs/fold-{self.fold_label}-{self.time_stamp}",
        )
        self.log_dir = os.path.join(
            output_dir, f"logs/fold-{self.fold_label}-{self.time_stamp}"
        )

    def fit_fold(
        self,
        loss,
        epochs,
        model_save_path,
        metric="mae",
        patience=10,
        early_stop_warmup=50,
        callbacks=[],
        lr=1e-4,
    ):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        optimizer = tf.keras.optimizers.AdamW(cos_decay_with_warmup(lr), beta_2=0.95)
        model_saver = SaveModel(model_save_path, 10, f"val_{metric}")
        core_callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                "val_loss", patience=patience, start_from_epoch=early_stop_warmup
            ),
            model_saver,
        ]
        self.model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)
        self.model.fit(
            self.train_data["dataset"],
            validation_data=self.val_data["dataset"],
            callbacks=[*callbacks, *core_callbacks],
            epochs=epochs,
            steps_per_epoch=self.train_data["steps_pre_epoch"],
            validation_steps=self.val_data["steps_pre_epoch"],
        )
        self.model.set_weights(model_saver.best_weights)
        self.metric_value = self.model.evaluate_metric(self.val_data["dataset"], metric)

    def save(self, path, save_format="keras"):
        self.model.save(path, save_format=save_format)

    def predict(self, dataset):
        return self.model.predict(dataset)


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, dataset):
        ensemble_preds = []
        for model in self.models:
            pred_val, _ = model.predict(dataset)
            ensemble_preds.append(pred_val)
        ensemble_pred = np.stack(ensemble_preds)
        ensemble_pred = np.reshape(ensemble_pred, newshape=[len(self.models), -1])
        ensemble_pred = np.mean(ensemble_pred, axis=0)
        return ensemble_pred

    def _find_best_model(self):
        best_model = self.models[0]
        best_mae = best_model.metric_value
        for model in self.models[1:]:
            if best_mae > model.metric_value:
                best_model = model
                best_mae = model.metric_value
        self.best_model = best_model

    def save_best_model(self, best_model_path):
        self._find_best_model()
        self.best_model.save(best_model_path, save_format="keras")

    def val_maes(self):
        self._find_best_model()
        maes = 0
        for model in self.models:
            maes += model.metric_value
        return self.best_model.metric_value, maes / len(self.models)

    def _mae(self, pred_val, true_val):
        abs = np.abs(true_val - pred_val)
        return np.mean(abs)

    def plot_fn(self, plot_fn, dataset, figure_dir, labels=None, is_category=False):
        self._find_best_model()

        best_figure_path = os.path.join(figure_dir, "best-model-test.png")
        ensemble_figure_path = os.path.join(figure_dir, "ensemble-model-test.png")
        best_pred, true_val = self.best_model.predict(dataset)

        ensemble_preds = []
        for model in self.models:
            model_pred, _ = model.predict(dataset)
            plot_fn(
                model_pred,
                true_val,
                os.path.join(figure_dir, f"model-{model.fold_label}-validation.png"),
                labels=labels,
            )
            ensemble_preds.append(model_pred)
        ensemble_pred = np.stack(ensemble_preds)
        ensemble_pred = np.reshape(ensemble_pred, newshape=[len(self.models), -1])
        if is_category:
            ensemble_pred = np.max(ensemble_pred, axis=0)
        else:
            ensemble_pred = np.mean(ensemble_pred, axis=0)

        plot_fn(best_pred, true_val, best_figure_path, labels=labels)
        plot_fn(ensemble_pred, true_val, ensemble_figure_path, labels=labels)

        if is_category:
            return
        best_mae = self._mae(best_pred, true_val)
        ensemble_mae = self._mae(ensemble_pred, true_val)
        return best_mae, ensemble_mae
