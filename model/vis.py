from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


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
