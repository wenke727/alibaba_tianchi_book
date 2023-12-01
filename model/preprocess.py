import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from loguru import logger


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)


def reduce_mem_usage(df, verbose=True, float_precision='medium'):
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min >= 0:
                    if c_max < 2**8:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 2**16:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 2**32:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
            else:
                if float_precision == 'high':
                    df[col] = df[col].astype(np.float64)
                elif float_precision == 'medium':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                elif float_precision == 'low':
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    else:
                        df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.debug(f'Memory usage after optimization is: {end_mem:.2f} MB. Decreased by {(100 * (start_mem - end_mem) / start_mem):.1f}%')

    return df

def generate_dataframe_stats(df, missing_threshold=95):
    """
    Generate statistics for each column in a Pandas DataFrame and identify columns
    with only one unique value or high percentage of missing values.

    Args:
    df (pd.DataFrame): The DataFrame for which statistics are to be calculated.
    missing_threshold (float): The threshold percentage for considering a column to have high missing values.

    Returns:
    pd.DataFrame: A DataFrame containing the statistics for each column.
    dict: A dictionary containing columns with high missing values and columns with only one unique value.
    """
    stats = []
    columns_to_drop = {'high_missing': [], 'low_variance': []}

    for col in df.columns:
        num_unique = df[col].nunique()
        pct_missing = df[col].isnull().sum() * 100 / df.shape[0]

        # Add to statistics
        stats.append((
            col, 
            num_unique,          
            pct_missing,          
            df[col].value_counts(normalize=True, dropna=False).values[0] * 100, 
            df[col].dtype
        ))

        # Check for columns to drop
        if pct_missing >= missing_threshold:
            columns_to_drop['high_missing'].append(col)
            logger.warning(f"Column {col} has a high percentage of missing values ({pct_missing}%)")

        if num_unique == 1:
            columns_to_drop['low_variance'].append(col)
            logger.warning(f"Column {col} has only one unique value and may not be useful for analysis.")

    stats_df = pd.DataFrame(stats, columns=[
        'Feature', 
        'Unique_values',          
        'Percentage of missing values',          
        'Percentage of values in the biggest category', 
        'Type'
    ])

    return stats_df, columns_to_drop

def plot_correlation_heatmap(df, show_lower_triangle=True,*args, **kwargs):
    corrmat = df.corr()
    plt.figure(figsize=(20, 9))
    mask=None
    if show_lower_triangle:
        mask = np.triu(np.ones_like(corrmat, dtype=bool), k=0)
    sns.heatmap(corrmat, mask=mask, square=True, *args, **kwargs)
    plt.show()
    
    return corrmat

def find_highly_correlated_columns(df, threshold=0.99):
    corr = df.corr().abs()
    n = len(df.columns)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(corr.iloc[i, j]) >= threshold:
                uf.union(i, j)

    # Creating groups
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(df.columns[i])

    # Determining columns to drop (all but the first in each group)
    _dict = {(k, v[0]): v[1:] for k, v in groups.items() if len(v) > 1}
    logger.debug(f"columns_to_drop: {_dict}")
    columns_to_drop = [col for group in groups.values() for col in group[1:]]

    return columns_to_drop

def analyze_and_identify_correlated_columns(df, threshold=0.99, plt_cfg={}):
    """
    Analyze the given DataFrame, plot a correlation heatmap, and identify columns 
    that are highly correlated and should be considered for removal.

    Args:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    list: A list of column names that are highly correlated and might be dropped.
    """
    # Selecting only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Plotting correlation heatmap
    plot_correlation_heatmap(df_numeric, **plt_cfg)

    # Finding and returning columns to drop
    columns_to_drop = find_highly_correlated_columns(df_numeric, threshold=threshold)

    return columns_to_drop


if __name__ == "__main__":
    # Load your DataFrame here
    # Example: df = pd.read_csv('path_to_your_data.csv')

    # Example DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1, 2, 3, 4, 5],
        'C': [5, 4, 6, 8, 1],
        'D': [31, 3, 5, 7, 9],
        'E': ['x', 'y', 'z', 'w', 'v']  # Non-numeric column
    })

    columns_to_drop = analyze_and_identify_correlated_columns(df)
    print("Columns to drop:", columns_to_drop)


    """ generate_dataframe_stats """
    # Example usage
    # train = pd.read_csv('your_data.csv')
    # stats_df, columns_to_drop = generate_dataframe_stats(train)
    # print(stats_df)
    # print(columns_to_drop)

