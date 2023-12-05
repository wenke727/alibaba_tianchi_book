import math
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from loguru import logger


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


def visualize_model_confusion_matrix(model, X, y, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
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
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

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
            f"{cm[i, j]}\n{cm_normalized[i, j]:.1f}%",
            horizontalalignment="center",
            color="white" if cm[i, j] > cm.max() / 2.0 else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    return fig, cm
