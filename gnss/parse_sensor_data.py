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

# step 3: 卫星覆盖程度
# df_converage = get_sat_coverage_dataframe(df_sat, ['rid'])
# df_converage = df_converage.merge(df[[LABEL]], left_index=True, right_index=True)
# sns.pairplot(data=df_converage, hue=LABEL)

# plot_satellite_distribution_seaborn(df_sat.query('rid==99'))
# df_sat.query('rid==99').to_csv('./sample.csv', index=False)



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

feats, _ = aggregate_data(
    group_cols=['rid'], 
    attrs=['satDb'], 
    # bin_att='satDb',
    # intvl=10,
    desc="haha",
    agg_ops=['count', 'mean', 'std', '50%', '75%'],
    # bin_att='satDb', intvl=10,
    **params
) 

feats
# np.allclose(feat_lst[0].values, feat_lst[1].values)

#%%
aggregate_data(df_sat, ['rid'], ['satDb'],  n_partitions=1, feat_lst=feat_lst, tag_2_feats=tag_2_feats, desc="haha") # bin_att='satDb', intvl=10,
feat_lst[0]
