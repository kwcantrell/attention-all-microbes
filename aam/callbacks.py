import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


def _mean_absolute_error(pred_val, true_val, fname, labels=None):
    pred_val = np.squeeze(pred_val)
    true_val = np.squeeze(true_val)
    mae = np.mean(np.abs(true_val - pred_val))

    min_x = np.min(true_val)
    max_x = np.max(true_val)
    coeff = np.polyfit(true_val, pred_val, deg=1)
    p = np.poly1d(coeff)
    xx = np.linspace(min_x, max_x, 50)
    yy = p(xx)

    diag = np.polyfit(true_val, true_val, deg=1)
    p = np.poly1d(diag)
    diag_xx = np.linspace(min_x, max_x, 50)
    diag_yy = p(diag_xx)
    data = {"pred": pred_val, "true": true_val}
    data = pd.DataFrame(data=data)
    plot = sns.scatterplot(data, x="true", y="pred")
    plt.plot(xx, yy)
    plt.plot(diag_xx, diag_yy)
    mae = "%.4g" % mae
    plot.set(xlabel="True")
    plot.set(ylabel="Predicted")
    plot.set(title=f"MAE: {mae}")
    plt.savefig(fname)
    plt.close()


def _confusion_matrix(pred_val, true_val, fname, cat_labels=None):
    cf_matrix = tf.math.confusion_matrix(true_val, pred_val).numpy()
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
    ]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(cf_matrix.shape)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(
        cf_matrix,
        annot=labels,
        xticklabels=cat_labels,
        yticklabels=cat_labels,
        fmt="",
    )
    import textwrap

    def wrap_labels(ax, width, break_long_words=False):
        labels = []
        for label in ax.get_xticklabels():
            text = label.get_text()
            labels.append(
                textwrap.fill(text, width=width, break_long_words=break_long_words)
            )
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(labels, rotation=0)

    wrap_labels(ax, 10)
    plt.savefig(fname)
    plt.close()


class MeanAbsoluteError(tf.keras.callbacks.Callback):
    def __init__(self, dataset, output_dir, report_back, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.output_dir = output_dir
        self.report_back = report_back

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.report_back == 0:
            y_pred, y_true = self.model.predict(self.dataset)
            _mean_absolute_error(y_pred, y_true, self.output_dir)


class ConfusionMatrx(tf.keras.callbacks.Callback):
    def __init__(self, dataset, output_dir, report_back, labels, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.report_back = report_back
        self.dataset = dataset
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        y_pred, y_true = self.model.predict(self.dataset)
        _confusion_matrix(y_pred, y_true, self.output_dir, self.labels)


class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, output_dir, report_back, monitor="val_loss", **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.report_back = report_back
        self.best_weights = None
        self.best_metric = None
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        metric = logs[self.monitor]
        if self.best_weights is None or self.best_metric > metric:
            self.best_metric = metric
            self.best_weights = self.model.get_weights()

        if epoch % self.report_back == 0:
            self.model.save(
                self.output_dir,
                save_format="keras",
            )

        logs["lr"] = self.model.optimizer.lr

    def get_config(self):
        base_config = super().get_config()
        config = {"output_dir": self.output_dir, "report_back": self.report_back}
        return {**base_config, **config}
