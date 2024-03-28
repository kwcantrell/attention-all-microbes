import os
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sepsis.metrics import denormalize


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def mean_absolute_error(dataset, model, fname, epoch, mean, std):
    pred_val = tf.squeeze(model.predict(dataset)).numpy()
    true_val = np.concatenate([tf.squeeze(ys).numpy() for (_, ys) in dataset])
    pred_val = denormalize(pred_val, mean, std)
    true_val = denormalize(true_val, mean, std)
    mae, h = mean_confidence_interval(np.abs(true_val - pred_val))

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

    plt.figure(figsize=(4, 4))
    plt.subplot(1, 1, 1)
    plt.scatter(true_val, pred_val, 7, marker='.', c='grey', alpha=0.5)
    plt.plot(xx, yy)
    plt.plot(diag_xx, diag_yy)
    mae, h = '%.4g' % mae, '%.4g' % h
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title(f"MAE: {mae}  {h} (epoch: {epoch})")
    plt.savefig(fname)
    plt.close()


def mae_scatter(
        mean,
        std,
        title,
        dataset,
        out_dir):
    class MAE_Scatter(tf.keras.callbacks.Callback):
        def __init__(self, title, dataset, out_dir, report_back_after_epochs=5):
            super().__init__()
            self.title = title
            self.dataset = dataset
            self.out_dir = out_dir
            self.report_back_after_epochs = report_back_after_epochs

        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.report_back_after_epochs == 0:
                mean_absolute_error(
                    self.dataset,
                    self.model,
                    fname=os.path.join(
                        self.out_dir, f'MAE-{self.title}-epoch-{epoch}.png'
                        ),
                    epoch=epoch,
                    mean=mean,
                    std=std
                )
            return super().on_epoch_end(epoch, logs)
    return MAE_Scatter(title, dataset, out_dir)
