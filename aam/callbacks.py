import os
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from aam.losses import _pairwise_distances
from skbio.stats.distance import DistanceMatrix
import skbio.stats.ordination
from biom import load_table
from aam.data_utils import (
    get_sequencing_dataset, get_unifrac_dataset, combine_datasets,
    batch_dataset,
)
from unifrac import unweighted


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def mean_absolute_error(mean, std, dataset, model, fname, epoch):
    pred_val = tf.squeeze(model.predict(dataset)).numpy()*std + mean
    true_val = np.concatenate([tf.squeeze(ys).numpy() for (_, ys) in dataset])
    true_val = true_val*std + mean
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


class MAE_Scatter(tf.keras.callbacks.Callback):
    def __init__(
        self,
        mean,
        std,
        title,
        dataset,
        out_dir,
        report_back_after_epochs=5
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.title = title
        self.dataset = dataset
        self.out_dir = out_dir
        self.report_back_after_epochs = report_back_after_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.report_back_after_epochs == 0:
            mean_absolute_error(
                self.mean,
                self.std,
                self.dataset,
                self.model,
                fname=os.path.join(
                    self.out_dir, f'MAE-{self.title}-epoch-{epoch}.png'
                    ),
                epoch=epoch
            )
        return super().on_epoch_end(epoch, logs)


class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, output_dir, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(
            os.path.join(self.output_dir, 'model.keras'),
            save_format='keras'
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "output_dir": self.output_dir
        }
        return {**base_config, **config}


class ProjectEncoder(tf.keras.callbacks.Callback):
    def __init__(self,
                 i_table,
                 i_tree,
                 output_dir,
                 batch_size):
        super().__init__()
        self.i_table = i_table
        self.i_tree = i_tree
        self.batch_size = batch_size
        self.table = load_table(i_table)
        self.output_dir = output_dir
        self.cur_step = 0
        self.num_samples = 250
        seq_dataset = get_sequencing_dataset(i_table)
        unifrac_dataset = get_unifrac_dataset(i_table, i_tree)
        dataset = combine_datasets(seq_dataset,
                                   unifrac_dataset,
                                   100,
                                   add_index=True)
        self.dataset = batch_dataset(dataset,
                                     batch_size,
                                     shuffle=False,
                                     repeat=1,
                                     is_pairwise=True)

    def _log_epoch_data(self):
        tf.print('loggin data...')
        total_samples = (int(self.table.shape[1] / self.batch_size)
                         * self.batch_size)

        sample_indices = np.arange(total_samples)
        np.random.shuffle(sample_indices)
        sample_indices = sample_indices[:self.num_samples]
        pred = self.model.predict(self.dataset)

        pred = tf.gather(pred, sample_indices)
        distances = _pairwise_distances(pred, squared=False)
        pred_unifrac_distances = DistanceMatrix(
            distances.numpy(),
            self.table.ids(axis='sample')[sample_indices],
            validate=False
        )
        pred_pcoa = skbio.stats.ordination.pcoa(pred_unifrac_distances,
                                                method='fsvd',
                                                number_of_dimensions=3,
                                                inplace=True)
        pred_pcoa.write(os.path.join(self.output_dir, 'pred_pcoa.pcoa'))

        true_unifrac_distances = unweighted(
                self.i_table, self.i_tree
                ).filter(self.table.ids(axis='sample')[sample_indices])
        true_pcoa = skbio.stats.ordination.pcoa(true_unifrac_distances,
                                                method='fsvd',
                                                number_of_dimensions=3,
                                                inplace=True)
        true_pcoa.write(os.path.join(self.output_dir, 'true_pcoa.pcoa'))

    def on_epoch_end(self, epoch, logs=None):
        if self.cur_step % 5 == 0:
            self._log_epoch_data()
            self.cur_step = 0
        self.cur_step += 1


class Accuracy(tf.keras.callbacks.Callback):
    def __init__(self, title, dataset, out_dir):
        super().__init__()
        self.title = title
        self.dataset = dataset
        self.out_dir = out_dir

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            pred_cat = tf.squeeze(self.model.predict(self.dataset)).numpy()
            true_cat = np.concatenate([tf.squeeze(ys).numpy() for (_, ys) in
                                       self.dataset])

            def plot_prc(name, labels, predictions, **kwargs):
                precision, recall, _ = sklearn.metrics.precision_recall_curve(
                    labels,
                    predictions
                )

                plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
                plt.xlabel('Precision')
                plt.ylabel('Recall')
                plt.grid(True)
                ax = plt.gca()
                ax.set_aspect('equal')
                fname = os.path.join(self.out_dir,
                                     f'auc-{epoch}.png')
                plt.savefig(fname)
                plt.close('all')

            plot_prc('AUC', true_cat, pred_cat)

            def plot_roc(name, labels, predictions, **kwargs):
                fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

                plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
                plt.xlabel('False positives [%]')
                plt.ylabel('True positives [%]')
                plt.xlim([-0.5, 20])
                plt.ylim([80, 100.5])
                plt.grid(True)
                ax = plt.gca()
                ax.set_aspect('equal')
                fname = os.path.join(self.out_dir,
                                     f'roc-{self.epoch}.png')
                plt.savefig(fname)
            plot_roc('ROC', true_cat, pred_cat)
        return super().on_epoch_end(epoch, logs)
