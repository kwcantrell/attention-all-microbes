import os
import scipy
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from aam.losses import denormalize


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def mean_absolute_error(dataset, y, hue, hue_label, model, fname, mean, std):
    pred_val = tf.reshape(model.predict(dataset), shape=[-1]).numpy()
    true_val = np.array(y, dtype=np.float32)
    pred_val = denormalize(pred_val, mean, std)
    mae, h = mean_confidence_interval(np.abs(pred_val - true_val))

    min_x = np.min(true_val)
    max_x = np.max(true_val)
    coeff = np.polyfit(true_val, pred_val, deg=1)
    p = np.poly1d(coeff)
    xx = np.linspace(min_x, max_x, 1000)
    yy = p(xx)

    diag = np.polyfit(true_val, true_val, deg=1)
    p = np.poly1d(diag)
    diag_xx = np.linspace(min_x, max_x, 1000)
    diag_yy = p(diag_xx)
    data = {"pred": pred_val, "true": true_val}
    if hue is not None:
        hue_label = hue_label if hue_label else 'groups'
        data[hue_label] = hue
    data = pd.DataFrame(data=data)
    plot = sns.scatterplot(data, x="true", y="pred", hue=hue_label)
    plt.plot(xx, yy)
    plt.plot(diag_xx, diag_yy)
    mae, h = '%.4g' % mae, '%.4g' % h
    plot.set(xlabel='True')
    plot.set(ylabel='Predicted')
    plot.set(title=f"MAE: {mae}  {h}")
    plt.savefig(fname)
    plt.close()

def violinplot(dataset, y, y_label, group, group_label, model, fname, mean, std):
    pred_val = tf.reshape(model.predict(dataset), shape=[-1]).numpy()
    true_val = np.array(y, dtype=int)
    pred_val = denormalize(pred_val, mean, std)
    category = ['predicted']*len(pred_val) + ['true']*len(true_val)
    variable = np.concatenate([pred_val.astype(int), true_val.astype(int)])
    group = np.concatenate([group, group])
    data = {f"{y_label}": variable, f"{group_label}": group, "category": category}
    data = pd.DataFrame(data=data)
    plot = sns.violinplot(data, x=f"{group_label}", y=f"{y_label}", hue="category")
    plot.set(title=f"Age Prediction")
    plt.savefig(fname)
    plt.close()


def violinplot_residuals(dataset, y, y_label, group, group_label, model, fname, mean, std):
    pred_val = tf.reshape(model.predict(dataset), shape=[-1]).numpy()
    true_val = np.array(y, dtype=int)
    pred_val = denormalize(pred_val, mean, std)
    residuals = pred_val - true_val
    max_val = np.max(true_val)
    bin_size = int(max_val / 5)
    true_val = pd.cut(np.array(y, dtype=int),
                      5#   [i for i in range(0, max_val, bin_size)],
    )
                    #   labels=[str(f'{i}-{i+bin_size}') for i in range(0, max_val, bin_size)])
    data = {f"{y_label}": residuals, "x": true_val, f"{group_label}": group}
    data = pd.DataFrame(data=data)
    plot = sns.boxplot(data, x="x", hue=f"{group_label}", y=f"{y_label}", showfliers=False)
    plot.set(title=f"Age Residuals")
    plt.savefig(fname)
    plt.close()


class MAE_Scatter(tf.keras.callbacks.Callback):
    def __init__(self,
                 title,
                 dataset,
                 metadata,
                 y_col,
                 hue_col=None,
                 hue_label=None,
                 mean=None,
                 std=None,
                 out_dir='',
                 report_back_after=5):
        super().__init__()
        self.mean=mean
        self.std=std
        self.title = title
        self.dataset = dataset
        self.metadata = metadata
        self.y = metadata[y_col].to_list()
        if hue_col:
            self.hue = metadata[hue_col].to_list()
        else:
            self.hue  = None
        self.hue_label = hue_label
        self.out_dir = out_dir
        self.report_back_after_epochs = report_back_after
        

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.report_back_after_epochs == 0:
            mean_absolute_error(
                self.dataset,
                self.y,
                self.hue,
                self.hue_label,
                self.model,
                fname=os.path.join(
                    self.out_dir, f'MAE-{self.title}.png'
                    ),
                mean=self.mean,
                std=self.std
            )
        return super().on_epoch_end(epoch, logs)

class ViolinPrediction(tf.keras.callbacks.Callback):
    def __init__(self,
                 title,
                 dataset,
                 metadata,
                 y_col,
                 y_label=None,
                 hue_col=None,
                 hue_label=None,
                 mean=None,
                 std=None,
                 out_dir='',
                 report_back_after=5):
        super().__init__()
        self.title = title
        self.dataset = dataset
        self.metadata = metadata
        self.y = metadata[y_col].to_list()
        if hue_col:
            self.hue = metadata[hue_col].to_list()
        else:
            self.hue  = None
        self.y_label = y_label
        self.hue_label = hue_label
        self.mean=mean
        self.std=std
        self.out_dir = out_dir
        self.report_back_after_epochs = report_back_after

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.report_back_after_epochs == 0:
            violinplot(
                self.dataset,
                self.y,
                self.y_label,
                self.hue,
                self.hue_label,
                self.model,
                fname=os.path.join(
                    self.out_dir, f'Violin-{self.title}.png'
                    ),
                mean=self.mean,
                std=self.std
            )
        return super().on_epoch_end(epoch, logs)


class ViolinResiduals(tf.keras.callbacks.Callback):
    def __init__(self,
                 title,
                 dataset,
                 metadata,
                 y_col,
                 y_label=None,
                 hue_col=None,
                 hue_label=None,
                 mean=None,
                 std=None,
                 out_dir='',
                 report_back_after=5):
        super().__init__()
        self.title = title
        self.dataset = dataset
        self.metadata = metadata
        self.y = metadata[y_col].to_list()
        if hue_col:
            self.hue = metadata[hue_col].to_list()
        else:
            self.hue  = None
        self.y_label = y_label
        self.hue_label = hue_label
        self.mean=mean
        self.std=std
        self.out_dir = out_dir
        self.report_back_after_epochs = report_back_after

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.report_back_after_epochs == 0:
            violinplot_residuals(
                self.dataset,
                self.y,
                self.y_label,
                self.hue,
                self.hue_label,
                self.model,
                fname=os.path.join(
                    self.out_dir, f'Violin-{self.title}.png'
                    ),
                mean=self.mean,
                std=self.std
            )
        return super().on_epoch_end(epoch, logs)
