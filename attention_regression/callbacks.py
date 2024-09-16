import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tensorflow as tf


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


@tf.keras.saving.register_keras_serializable(package="mean_absolute_error")
def mean_absolute_error(dataset, model, fname):
    pred_val, true_val = model.predict(dataset)
    pred_val = tf.reshape(pred_val, shape=[-1]).numpy()
    true_val = tf.reshape(true_val, shape=[-1]).numpy()

    pred_val = pred_val.astype(np.float32)
    true_val = true_val.astype(np.float32)
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


@tf.keras.saving.register_keras_serializable(package="mean_absolute_error")
def confusion_matrix(dataset, model, fname, cat_labels=None):
    y_pred, y_true = model.predict(dataset)
    cf_matrix = tf.math.confusion_matrix(y_true, y_pred).numpy()
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
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position("top")
    import textwrap

    # ax.set_xticklabels(labels, rotation=0)
    # plt.xticks(rotation=45)
    # plt.yticks(rotation=0)

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


def violinplot(dataset, y, y_label, group, group_label, model, fname, mean, std):
    pred_val = tf.reshape(model.predict(dataset), shape=[-1]).numpy()
    true_val = np.array(y, dtype=int)
    pred_val = denormalize(pred_val, mean, std)
    category = ["predicted"] * len(pred_val) + ["true"] * len(true_val)
    variable = np.concatenate([pred_val.astype(int), true_val.astype(int)])
    group = np.concatenate([group, group])
    data = {f"{y_label}": variable, f"{group_label}": group, "category": category}
    data = pd.DataFrame(data=data)
    plot = sns.violinplot(data, x=f"{group_label}", y=f"{y_label}", hue="category")
    plot.set(title="Age Prediction")
    plt.savefig(fname)
    plt.close()


def violinplot_residuals(
    dataset, y, y_label, group, group_label, model, fname, mean, std
):
    pred_val = tf.reshape(model.predict(dataset), shape=[-1]).numpy()
    true_val = np.array(y, dtype=int)
    pred_val = denormalize(pred_val, mean, std)
    residuals = pred_val - true_val
    true_val = pd.cut(np.array(y, dtype=int), 5)
    data = {f"{y_label}": residuals, "x": true_val, f"{group_label}": group}
    data = pd.DataFrame(data=data)
    plot = sns.boxplot(
        data, x="x", hue=f"{group_label}", y=f"{y_label}", showfliers=False
    )
    plot.set(title="Age Residuals")
    plt.savefig(fname)
    plt.close()


def feature_confidences(tokens, feature_embeddings):
    total_tokens = tf.reduce_max(tokens)
    true_counts = np.zeros(total_tokens + 1, dtype=np.float32)
    false_counts = np.zeros(total_tokens + 1, dtype=np.float32)
    true_conf = np.zeros(total_tokens + 1, dtype=np.float32)
    false_conf = np.zeros(total_tokens + 1, dtype=np.float32)

    tokens = tokens.numpy()
    feature_class = tf.keras.activations.softmax(feature_embeddings)
    feature_class = feature_class.numpy()

    for s_tokens, s_conf in zip(tokens, feature_class):
        tf.print(tf.shape(s_tokens))
        token_mask = s_tokens > 0
        valid = np.sum(token_mask)
        true_indices = s_tokens[token_mask]
        true_counts[true_indices] += 1

        false_indices = s_tokens[~token_mask]
        false_counts[false_indices] += 1

        true_conf[true_indices] += s_conf[:valid, 1]
        false_conf[false_indices] += s_conf[valid:, 0]

    true_count_mask = true_counts > 0
    true_conf[true_count_mask] /= true_counts[true_count_mask]
    false_count_mask = false_counts > 0
    false_conf[false_count_mask] /= false_counts[false_count_mask]

    true_tokens = np.argwhere(true_count_mask)
    true_tokens = np.reshape(true_tokens, newshape=-1)
    in_pd = pd.DataFrame(
        {"Confidence": true_conf[true_count_mask], "Token": true_tokens}
    )
    in_pd["Class"] = "Features in Sample"

    false_tokens = np.argwhere(false_count_mask)
    false_tokens = np.reshape(false_tokens, newshape=-1)
    not_pd = pd.DataFrame(
        {"Confidence": false_conf[false_count_mask], "Token": false_tokens}
    )
    not_pd["Class"] = "Features not in Sample"
    data = pd.concat((in_pd, not_pd))
    return data


@tf.keras.saving.register_keras_serializable(package="MAE_Scatter")
class MAE_Scatter(tf.keras.callbacks.Callback):
    def __init__(
        self,
        title,
        dataset,
        out_dir="",
        report_back_after=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title = title
        self.dataset = dataset
        self.out_dir = out_dir
        self.report_back_after_epochs = report_back_after

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.report_back_after_epochs == 0:
            mean_absolute_error(
                self.dataset,
                self.model,
                fname=os.path.join(self.out_dir, f"MAE-{self.title}.png"),
            )
        return super().on_epoch_end(epoch, logs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "title": self.title,
            "dataset": self.dataset,
            "out_dir": self.out_dir,
            "report_back_after": self.report_back_after,
        }
        return {**base_config, **config}


@tf.keras.saving.register_keras_serializable(package="ConfusionMatrix")
class ConfusionMatrix(tf.keras.callbacks.Callback):
    def __init__(
        self,
        title,
        dataset,
        cat_labels=None,
        out_dir="",
        report_back_after=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title = title
        self.dataset = dataset
        self.cat_labels = cat_labels
        self.out_dir = out_dir
        self.report_back_after_epochs = report_back_after

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.report_back_after_epochs == 0:
            confusion_matrix(
                self.dataset,
                self.model,
                fname=os.path.join(self.out_dir, f"{self.title}.png"),
                cat_labels=self.cat_labels,
            )
        return super().on_epoch_end(epoch, logs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "title": self.title,
            "dataset": self.dataset,
            "cat_labels": self.cat_labels,
            "out_dir": self.out_dir,
            "report_back_after": self.report_back_after,
        }
        return {**base_config, **config}


class ViolinPrediction(tf.keras.callbacks.Callback):
    def __init__(
        self,
        title,
        dataset,
        metadata,
        y_col,
        y_label=None,
        hue_col=None,
        hue_label=None,
        mean=None,
        std=None,
        out_dir="",
        report_back_after=5,
    ):
        super().__init__()
        self.title = title
        self.dataset = dataset
        self.metadata = metadata
        self.y = metadata[y_col].to_list()
        if hue_col:
            self.hue = metadata[hue_col].to_list()
        else:
            self.hue = None
        self.y_label = y_label
        self.hue_label = hue_label
        self.mean = mean
        self.std = std
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
                fname=os.path.join(self.out_dir, f"Violin-{self.title}.png"),
                mean=self.mean,
                std=self.std,
            )
        return super().on_epoch_end(epoch, logs)


class ViolinResiduals(tf.keras.callbacks.Callback):
    def __init__(
        self,
        title,
        dataset,
        metadata,
        y_col,
        y_label=None,
        hue_col=None,
        hue_label=None,
        mean=None,
        std=None,
        out_dir="",
        report_back_after=5,
    ):
        super().__init__()
        self.title = title
        self.dataset = dataset
        self.metadata = metadata
        self.y = metadata[y_col].to_list()
        if hue_col:
            self.hue = metadata[hue_col].to_list()
        else:
            self.hue = None
        self.y_label = y_label
        self.hue_label = hue_label
        self.mean = mean
        self.std = std
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
                fname=os.path.join(self.out_dir, f"Violin-{self.title}.png"),
                mean=self.mean,
                std=self.std,
            )
        return super().on_epoch_end(epoch, logs)


@tf.keras.saving.register_keras_serializable(package="AvgFeatureConfidence")
class AvgFeatureConfidence(tf.keras.callbacks.Callback):
    def __init__(
        self,
        title,
        dataset,
        metadata,
        y_col,
        y_label=None,
        hue_col=None,
        hue_label=None,
        out_dir="",
        report_back_after=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title = title
        self.dataset = dataset
        self.metadata = metadata
        self.y_col = y_col
        self.y_label = y_label
        self.hue_col = hue_col
        self.hue_label = hue_label
        self.out_dir = out_dir
        self.report_back_after_epochs = report_back_after

        self.y = metadata[y_col].to_list()
        if hue_col:
            self.hue = metadata[hue_col].to_list()
        else:
            self.hue = None

    def on_epoch_end(self, epoch, logs=None):
        # if epoch % self.report_back_after_epochs == 0:
        #     plot_feature_confidence(
        #         self.dataset,
        #         self.y,
        #         self.y_label,
        #         self.hue,
        #         self.hue_label,
        #         self.model,
        #         fname=os.path.join(
        #             self.out_dir, f'confidence-{self.title}.png'
        #             ),
        #     )
        self.model.feature_confidences(self.dataset)
        return super().on_epoch_end(epoch, logs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "title": self.title,
            "dataset": self.dataset,
            "metadata": self.metadata,
            "y_col": self.y_col,
            "y_label": self.y_label,
            "hue_col": self.hue_col,
            "hue_label": self.hue_label,
            "out_dir": self.out_dir,
            "report_back_after_epochs": self.report_back_after,
        }
        return {**base_config, **config}
