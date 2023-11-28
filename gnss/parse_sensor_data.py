#%%
import sys
sys.path.append('../')
import os
import json
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from loguru import logger
from scipy.optimize import linear_sum_assignment

from cfg import DATA_FOLDER
from feature import get_sat_coverage_dataframe
from vis import plot_satellite_distribution_seaborn
from model.preprocess import reduce_mem_usage
from model.featureEng import aggregate_data

import warnings
warnings.filterwarnings('ignore')

KEYWORD_2_COLUMN = {
    'light': ["values"], 
    'mobile': ['cid'],
}

LABEL = "label"

""" 工具函数 """
def delete_cache_gnss_records(data):
    ori_size = data.shape[0]
    data.isUsed = data.isUsed.astype(bool)
    _df_sat = data.query("satDb > 0 or isUsed == True")
    cur_size = _df_sat.shape[0]
    logger.debug(f"Delete {ori_size - cur_size} records, remaine {cur_size} records , {(ori_size - cur_size) / ori_size * 100:.1f}% off")

    return _df_sat

def get_label(df, fn):
    df.loc[:, LABEL] = df[fn].apply(lambda x: True if 'in' in x else(False if 'out' in x else np.nan))
    
    return df

def group_kfold_split(X, y, groups, n_splits=10):
    """
    Split the dataset into training and validation sets using GroupKFold.

    This function is a generator that yields one split of the data at a time.
    It returns the training and validation sets for features (X) and targets (y).

    :param X: Features (numpy array or pandas DataFrame)
    :param y: Targets (numpy array or pandas Series)
    :param groups: Group labels (numpy array or pandas Series)
    :param n_splits: Number of folds. Must be at least 2. (default: 10)
    :return: A generator, which yields the training and validation sets for each fold
    """
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)

    # Iterate over each split
    for train_idx, valid_idx in gkf.split(X, y, groups):
        # Yield the subsets of the data for the current split
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        yield X_train, y_train, X_valid, y_valid


""" 读取函数 """
def convert_timestamp_to_datetime_custom_unit(df: pd.DataFrame, column_name: str, unit: str = 'ms') -> pd.DataFrame:
    """
    Convert a timestamp column in a DataFrame to a datetime column in UTC+8 timezone with a specified unit.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the timestamp column.
    - column_name (str): The name of the timestamp column to convert.
    - unit (str): The unit of the timestamp ('s', 'ms', 'us', etc.). Default is 'ms'.
    
    Returns:
    - pd.DataFrame: The DataFrame with the timestamp column converted to datetime in UTC+8 timezone.
    """
    df[column_name] = pd.to_datetime(df[column_name], unit=unit)
    
    if df[column_name].dt.tz is not None:
        df[column_name] = df[column_name].dt.tz_localize(None)
    
    df[column_name] = df[column_name].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    return df

def explode_sat_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the satDataList column of the DataFrame and return a new DataFrame with satellite data.
    
    Parameters:
    - df (pd.DataFrame): The original DataFrame containing the satDataList column.
    
    Returns:
    - pd.DataFrame: A DataFrame with satellite data.
    """
    empty_idxs = df.query("satDataList != satDataList").index
    logger.debug(f"There are {len(empty_idxs)} empty satDataList records.")
    
    df.query("satDataList == satDataList", inplace=True)
    
    # Explode the satDataList column
    exploded_df = df[['satDataList']].explode('satDataList').reset_index()
    
    # Extract the satellite data into separate columns
    sat_df = pd.json_normalize(exploded_df['satDataList'])
    
    # Add the original index as a column to ensure it can be matched with the original df
    sat_df['rid'] = exploded_df['index']
    
    return sat_df

def optimal_bipartite_matching(gnss_df, light_df):
    gnss_times = gnss_df['timestamp'].values#.dt.tz_localize(None).values.astype(np.int64)
    light_times = light_df['timestamp'].values#.dt.tz_localize(None).values.astype(np.int64)
    
    time_diff_matrix = np.abs(np.subtract.outer(gnss_times, light_times))
    
    gnss_indices, light_indices = linear_sum_assignment(time_diff_matrix)
    
    valid_matches = np.where(time_diff_matrix[gnss_indices, light_indices] <= 60 * 1e9)[0]  # 1e9 converts seconds to nanoseconds
    
    return gnss_indices[valid_matches], light_indices[valid_matches]

def extract_data_from_zip(zip_filename: str, keyword: str) -> pd.DataFrame:
    """
    Extracts data from a zip file based on a keyword in filenames and returns a DataFrame.
    
    Parameters:
    - zip_filename (str): The path to the zip file.
    - keyword (str): The keyword to search for in filenames.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the extracted data.
    """
    all_records = []

    with zipfile.ZipFile(zip_filename, 'r') as z:
        filenames = [name for name in z.namelist() if keyword in name and name.endswith(".txt")]
        
        for filename in filenames:
            with z.open(filename) as f:
                for line in f:
                    line = line.decode('utf-8').strip()  
                    if line:  
                        all_records.append(json.loads(line))

    # Convert the JSON records to a pandas DataFrame
    return pd.DataFrame(all_records)

def extract_data_from_directory(folder: str, keywords: str, unit: str = 'ms') -> pd.DataFrame:
    """
    Extract data from all zip files in a given directory based on a keyword in filenames and returns a DataFrame.
    
    Parameters:
    - directory_path (str): The path to the directory containing zip files.
    - keyword (str): The keyword to search for in filenames.
    - unit (str): The unit of the timestamp ('s', 'ms', 'us', etc.). Default is 'ms'.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the extracted data from all zip files.
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    assert "GNSS" in keywords, "Check the keywords"
    
    all_dataframes = []
    for filename in os.listdir(folder):
        if not filename.endswith(".zip"):
            continue
        
        logger.debug(f"process: {filename}")
        df = extract_data_from_zip(os.path.join(folder, filename), "GNSS")

        for i in keywords:
            if i == "GNSS":
                continue
            _df = extract_data_from_zip(os.path.join(folder, filename), i)
            left_idxs, right_idxs = optimal_bipartite_matching(df, _df)
            cols = KEYWORD_2_COLUMN[i]
            for col in cols:
                df[f"{i}_{col}"] = np.nan
                df.loc[left_idxs, f"{i}_{col}"] = _df.loc[right_idxs, col].values

        # df = convert_timestamp_to_datetime_custom_unit(df, 'timestamp', unit)
        
        df.loc[:, 'fn'] = filename
        df = get_label(df, 'fn')
        all_dataframes.append(df)

    return pd.concat(all_dataframes, ignore_index=True)


#%%
if __name__ == "__main__":
    df = extract_data_from_directory(DATA_FOLDER, ["GNSS", 'light', 'mobile'])
    df_sat = explode_sat_data(df)
    df_sat


# %%
# step 1: delete `NaN`` 
df_sat = reduce_mem_usage(df_sat)
df_sat.satFrequency = np.round((df_sat.satFrequency / 1e6).values, 2)

# step 2: delete `cache`
df_sat = delete_cache_gnss_records(df_sat)

#%%
#! max increase Lengh
# step 3: 卫星覆盖程度

def satellite_coverage_metric(df, elevation_interval=30, azimuth_interval=30):
    df['elevation_bin'] = (df['elevation'] // elevation_interval) * elevation_interval
    df['azimuth_bin'] = (df['azimuth'] // azimuth_interval) * azimuth_interval
    
    coverage_df = df.groupby(['elevation_bin', 'azimuth_bin']).agg({
        'satDb': ['count', 'mean']
    }).reset_index()
    
    coverage_df.columns = ['elevation_bin', 'azimuth_bin', 'sat_count', 'snr_mean']
    
    total_bins = (360 // azimuth_interval) * (90 // elevation_interval)
    occupied_bins = coverage_df['sat_count'].gt(0).sum()
    coverage_percentage = (occupied_bins / total_bins)
    average_snr = coverage_df['snr_mean'].mean()
    snr_std = coverage_df['snr_mean'].std()
    
    summary = {
        # 'total_bins': total_bins,
        # 'occupied_bins': occupied_bins,
        'elevation_bin': (coverage_df.elevation_bin.values // 30).astype(int),
        'azimuth_bin': (coverage_df.azimuth_bin.values // 30).astype(int),
        'coverage_percentage': coverage_percentage,
        'average_snr': average_snr,
        'snr_std_dev': snr_std
    }
    
    return summary

def get_sat_coverage_dataframe(df_sat, group_cols=['rid']):
    data = df_sat.groupby(group_cols).apply(satellite_coverage_metric)

    df_coverage = pd.json_normalize(data=data)
    df_coverage.index = data.index
    
    return df_coverage


df_converage = get_sat_coverage_dataframe(df_sat, ['rid'])
df_converage = df_converage.merge(df[[LABEL]], left_index=True, right_index=True)
df_converage

#%%

def longestConsecutive(nums, max_val=11, verbose=True):
    if verbose: 
        logger.debug(f"nums: {nums}")
    if not nums:
        return 0

    def dp_longest_consecutive(nums):
        nums = sorted(set(nums))  # Remove duplicates and sort
        dp = [1] * len(nums)
        max_len = 1 if nums else 0
        start = 0

        for i in range(1, len(nums)):
            if nums[i] - nums[i - 1] == 1:
                dp[i] = dp[i - 1] + 1
                if dp[i] > max_len:
                    max_len = dp[i]
                    start = i - dp[i] + 1
            else:
                dp[i] = 1

        end = start + max_len - 1
        return max_len, (nums[start], nums[end]) if max_len > 0 else None

    nums_adjusted = [num if num != 0 else max_val + 1 for num in nums]
    nums_adjusted += [num + max_val + 1 for num in nums if num < max_val]

    max_len_original, interval_original = dp_longest_consecutive(nums)
    max_len_adjusted, interval_adjusted = dp_longest_consecutive(nums_adjusted)

    if verbose:
        if max_len_original >= max_len_adjusted:
            logger.debug(f"Longest Length = {max_len_original}, Interval = {interval_original}")
        else:
            logger.debug(f"Longest Length = {max_len_adjusted}, Interval = {interval_adjusted}")

    return max(max_len_original, max_len_adjusted)

rid = 5357
elevation_bin = df_converage.loc[rid].elevation_bin
azimuth_bin = df_converage.loc[rid].azimuth_bin

mask = elevation_bin == 0
longestConsecutive(azimuth_bin[mask].tolist());

mask = elevation_bin == 1
longestConsecutive(azimuth_bin[mask].tolist());

mask = elevation_bin == 2
longestConsecutive(azimuth_bin[mask].tolist());

longestConsecutive(np.unique(azimuth_bin).tolist());

# longestConsecutive([7, 8, 9, 11, 0, 1])



# sns.pairplot(data=df_converage, hue=LABEL)

# plot_satellite_distribution_seaborn(df_sat.query('rid==99'))
# df_sat.query('rid==99')


# step ?: 切分数据
# X_train, y_train, X_valid, y_valid = next(group_kfold_split(
#     df_converage, df_converage[LABEL], df.loc[df_converage.index].fn))

# %%
#! 统计特征
feat_lst = []
tag_2_feats = {}
params = {
    'data': df_sat, 
    'feat_lst': feat_lst,
    'tag_2_feats': tag_2_feats, 
    'n_partitions': 4,
}

#%%
# 1. 总体 satDb 的统计
feats, _ = aggregate_data(
    group_cols=['rid'], 
    attrs=['satDb'], 
    desc="satDb",
    agg_ops=['count', 'mean', 'std', '50%', '75%'],
    **params
) 

# 2. GPS Beidou 的统计
feats, _ = aggregate_data(
    group_cols=['rid', 'satTye'], 
    attrs=['satDb'], 
    filter_sql="satTye in [1, 5]",
    desc="satTye+satDb",
    agg_ops=['count', 'mean', 'std', '50%', '75%'],
    **params
) 

feats

# %%

# 3. 不同类型的卫星个数
feats, _ = aggregate_data(
    group_cols=['rid', 'satTye'], 
    attrs=['satIdentification'], 
    desc="satTye count",
    agg_ops=['nunique', 'unique'], # 'count'
    **params
) 

feats
# feats.fillna(0).astype(np.int8)

# %%
# ! 4. GPS 分桶
feats, _ = aggregate_data(
    group_cols=['rid', 'satTye'], 
    attrs=['satDb'], 
    bin_att='satDb',
    intvl=10,
    filter_sql="satTye in [1]",
    desc="satTye + satDb + satBin",
    agg_ops=['count', 'mean'],
    **params
) 

feats
# %%
