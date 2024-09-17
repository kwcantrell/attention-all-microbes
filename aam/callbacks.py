import os

import matplotlib.pyplot as plt
import numpy as np
import skbio.stats.ordination
import sklearn
import tensorflow as tf
from biom import load_table
from skbio.stats.distance import DistanceMatrix

from aam.data_utils import get_unifrac_dataset
from aam.losses import _pairwise_distances


class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, output_dir, report_back, monitor, **kwargs):
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


class ProjectEncoder(tf.keras.callbacks.Callback):
    def __init__(self, i_table, i_tree, output_dir, batch_size):
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
        dataset = combine_datasets(seq_dataset, unifrac_dataset, 100, add_index=True)
        self.dataset = batch_dataset(
            dataset, batch_size, shuffle=False, repeat=1, is_pairwise=True
        )

    def _log_epoch_data(self):
        tf.print("loggin data...")
        total_samples = int(self.table.shape[1] / self.batch_size) * self.batch_size

        sample_indices = np.arange(total_samples)
        np.random.shuffle(sample_indices)
        sample_indices = sample_indices[: self.num_samples]
        pred = self.model.predict(self.dataset)

        pred = tf.gather(pred, sample_indices)
        distances = _pairwise_distances(pred, squared=False)
        pred_unifrac_distances = DistanceMatrix(
            distances.numpy(),
            self.table.ids(axis="sample")[sample_indices],
            validate=False,
        )
        pred_pcoa = skbio.stats.ordination.pcoa(
            pred_unifrac_distances, method="fsvd", number_of_dimensions=3, inplace=True
        )
        pred_pcoa.write(os.path.join(self.output_dir, "pred_pcoa.pcoa"))

        true_unifrac_distances = unweighted(self.i_table, self.i_tree).filter(
            self.table.ids(axis="sample")[sample_indices]
        )
        true_pcoa = skbio.stats.ordination.pcoa(
            true_unifrac_distances, method="fsvd", number_of_dimensions=3, inplace=True
        )
        true_pcoa.write(os.path.join(self.output_dir, "true_pcoa.pcoa"))

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
            true_cat = np.concatenate(
                [tf.squeeze(ys).numpy() for (_, ys) in self.dataset]
            )

            def plot_prc(name, labels, predictions, **kwargs):
                precision, recall, _ = sklearn.metrics.precision_recall_curve(
                    labels, predictions
                )

                plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
                plt.xlabel("Precision")
                plt.ylabel("Recall")
                plt.grid(True)
                ax = plt.gca()
                ax.set_aspect("equal")
                fname = os.path.join(self.out_dir, f"auc-{epoch}.png")
                plt.savefig(fname)
                plt.close("all")

            plot_prc("AUC", true_cat, pred_cat)

            def plot_roc(name, labels, predictions, **kwargs):
                fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

                plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
                plt.xlabel("False positives [%]")
                plt.ylabel("True positives [%]")
                plt.xlim([-0.5, 20])
                plt.ylim([80, 100.5])
                plt.grid(True)
                ax = plt.gca()
                ax.set_aspect("equal")
                fname = os.path.join(self.out_dir, f"roc-{self.epoch}.png")
                plt.savefig(fname)

            plot_roc("ROC", true_cat, pred_cat)
        return super().on_epoch_end(epoch, logs)
