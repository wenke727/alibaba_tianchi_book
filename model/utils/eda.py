import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

from loguru import logger


def compare_distributions(train_data, test_data, column):
    """
    This function compares the distributions of a specified column between training and test datasets using KDE plots.
    If the difference between distributions is significant, a warning is logged.
    
    Parameters:
    - train_data: a pandas DataFrame containing the training dataset.
    - test_data: a pandas DataFrame containing the test dataset.
    - column: a string representing the column name to compare between the datasets.
    """
    fig = plt.figure(figsize=(10, 5))
    
    # KDE plot for training data
    ax = sns.kdeplot(train_data[column], color="Red", shade=True)
    
    # KDE plot for test data
    sns.kdeplot(test_data[column], color="Blue", shade=True)
    
    # Set the labels and legend
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.legend(["Train", "Test"])
    
    plt.show()
    
    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(train_data[column], test_data[column])
    
    # Log the results
    if p_value < 0.05:
        logger.warning(f'Significant difference in distributions for {column} (p-value: {p_value}).')
    else:
        logger.info(f'No significant difference in distributions for {column} (p-value: {p_value}).')

    return fig

def plot_feature_relationship_and_distribution(data, feature, target, target_type='continuous'):
    """
    This function plots the relationship and distribution of a feature with respect to a target variable.
    It adapts the plots based on the target variable type (continuous or categorical).
    
    Parameters:
    - data: a pandas DataFrame containing the dataset.
    - feature: a string representing the feature column name.
    - target: a string representing the target column name.
    - target_type: a string indicating the target variable type ('continuous' or 'categorical').
    """
    fcols = 2
    frows = 1

    plt.figure(figsize=(8 * fcols, 4 * frows))

    # Relationship plot (scatter or boxplot)
    ax = plt.subplot(1, 2, 1)
    if target_type == 'continuous':
        sns.regplot(x=feature, y=target, data=data, ax=ax,
                    scatter_kws={'marker':'.','s':3,'alpha':0.3},
                    line_kws={'color':'k'})
    elif target_type == 'categorical':
        sns.boxplot(x=target, y=feature, data=data, ax=ax)
    plt.xlabel(feature)
    plt.ylabel(target)

    # Distribution plot
    ax = plt.subplot(1, 2, 2)
    sns.histplot(data[feature].dropna())
    plt.xlabel(feature)

    plt.tight_layout()
    plt.show()

def plot_joint_distribution(data, feature, target):
    """
    This function plots the joint distribution of a feature and a target variable.
    It shows the relationship between the two variables along with their individual distributions.
    
    Parameters:
    - data: a pandas DataFrame containing the dataset.
    - feature: a string representing the feature column name.
    - target: a string representing the target column name.
    绘制特征和标签的联合分布图可以帮助我们理解它们之间的关系，特别是当我们怀疑存在非线性关系时。seaborn 库中的 jointplot 是一个很好的工具，它可以在散点图的基础上添加直方图和核密度估计，以便同时查看单变量和双变量分布。
    """
    # Create a jointplot of the feature and the target variable
    g = sns.jointplot(x=feature, y=target, data=data, kind="reg",
                      joint_kws={'scatter_kws': {'alpha': 0.5}, 'line_kws': {'color': 'red'}})
    
    # Set the axis labels
    g.set_axis_labels(feature, target)
    
    plt.show()
