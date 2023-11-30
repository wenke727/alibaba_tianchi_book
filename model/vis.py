import math
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from loguru import logger


def plot_kde_grid(df, n_col, hue="label", n_row=None, *args, **kwargs):
    """
    对DataFrame中的数值列绘制核密度估计图。

    :param df: pandas DataFrame，包含数值型数据。
    :param n_col: 网格的列数。
    :param n_row: 网格的行数。如果为None，则会自动计算。
    :param hue: 用于在kdeplot中进行分组的列名，默认为'label'。
    """
    num_cols = sorted(df.select_dtypes(include=["number"]).columns)
    logger.debug(
        f"plot kde plot for: {num_cols}, except {[i for i in df.columns if i not in num_cols]}"
    )

    if n_row is None:
        n_row = math.ceil(len(num_cols) / n_col)

    # create axes
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 5, n_row * 3))
    if n_row * n_col > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, col in enumerate(num_cols):
        if i < n_row * n_col:
            sns.kdeplot(
                data=df,
                x=col,
                fill=True,
                hue=df[hue] if hue in df.columns else None,
                ax=axes[i],
                *args,
                **kwargs,
            )
            axes[i].set_title(col)
            axes[i].set_xlabel("")
        else:
            break

    # 隐藏多余的图表
    for j in range(i + 1, n_row * n_col):
        axes[j].axis("off")

    plt.tight_layout()
    # plt.show()

    return fig


def plot_qq_with_subfigures(dataframe, column):
    """
    Plot a Q-Q plot for the specified column in the given dataframe based on a condition
    using subfigures.

    Parameters:
    dataframe (DataFrame): The dataframe containing the data.
    condition (Series): A boolean Series indicating the condition to filter the dataframe.
    column (str): The column for which to plot the Q-Q plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Distribution plot
    sns.distplot(dataframe[column], fit=stats.norm, ax=axes[0])

    # Q-Q plot
    stats.probplot(dataframe[column], plot=axes[1])

    return fig


def visualize_model_confusion_matrix(model, X, y, classes, title="Confusion Matrix", cmap=plt.cm.Blues, normalize=False):
    """
    Trains a classifier, makes predictions, and plots the confusion matrix.
    The confusion matrix displays both the actual counts and the relative proportions, if normalization is True.

    Parameters:
    - model: The classifier to be used for prediction.
    - X: Feature data for making predictions.
    - y: Actual labels.
    - classes: List of class names for the labels.
    - title (str): Title of the confusion matrix plot.
    - cmap: Colormap for the matrix.
    - normalize (bool): If True, the confusion matrix is normalized.

    Returns:
    - tuple: A tuple containing the matplotlib figure object and the confusion matrix array.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Loop over data dimensions and create text annotations.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i,
            f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
            horizontalalignment="center",
            color="white" if cm[i, j] > cm.max() / 2.0 else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    return fig, cm