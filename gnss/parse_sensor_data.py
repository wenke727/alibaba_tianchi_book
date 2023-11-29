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
from model.vis import plot_kde_grid
from model.preprocess import reduce_mem_usage
from model.featureEng import aggregate_data
from model.preprocess import analyze_and_identify_correlated_columns

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

def append_label(feats, df):
    return feats.merge(df[[LABEL]], left_index=True, right_index=True)

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
df_converage = get_sat_coverage_dataframe(df_sat, ['rid'])
df_converage = append_label(df_converage, df)
plot_kde_grid(df_converage, hue=LABEL, n_col=3)
# sns.pairplot(data=df_converage, hue=LABEL)
df_converage

# plot_satellite_distribution_seaborn(df_sat.query('rid==99'))
# df_sat.query('rid==99')

#%%

# step ?: 切分数据
# X_train, y_train, X_valid, y_valid = next(group_kfold_split(
#     df_converage, df_converage[LABEL], df.loc[df_converage.index].fn))

# %%
#! 统计特征
def feature_pipline_for_GNSS():
    #%%
    feat_lst = []
    tag_2_feats = {'base': ['satCount', 'useSatCount']}
    params = {
        'data': df_sat, 
        'feat_lst': feat_lst,
        'tag_2_feats': tag_2_feats, 
        'n_partitions': 4,
    }

    """ 1. satDb 的统计 """
    # 1.1 whole
    feats, _ = aggregate_data(
        group_cols=['rid'], 
        attrs=['satDb'], 
        desc="satDb",
        agg_ops=['count', 'mean', 'std', '25%', '50%', '75%'],
        **params
    ) 
    feats = append_label(feats, df)
    plot_kde_grid(feats, hue=LABEL, n_col=3);

    #%%
    # 1.2 SatDbBin
    feats, _ = aggregate_data(
        group_cols=['rid'], 
        attrs=['satDb'],     
        bin_att='satDb',
        intvl=10,
        filter_sql="satTye in [1, 5]",
        desc="satTye",
        agg_ops=['count', 'mean', 'std', '50%', '75%'],
        **params
    ) 
    feats = append_label(feats, df)
    plot_kde_grid(feats, hue=LABEL, n_col=5, common_norm=False);
    feats

    #%%
    # 1.3 GNSS Type
    feats, _ = aggregate_data(
        group_cols=['rid', 'satTye'], 
        attrs=['satDb'], 
        filter_sql="satTye in [1, 5]",
        desc="satTye",
        agg_ops=['count', 'mean', 'std', '50%', '75%'],
        **params
    ) 
    feats = append_label(feats, df)
    plot_kde_grid(feats, hue=LABEL, n_col=5, common_norm=False);
    feats

    # %%
    # 1.4 GNSS Type + SatDbBin
    feats, _ = aggregate_data(
        group_cols=['rid', 'satTye'], 
        attrs=['satDb'], 
        bin_att='satDb',
        intvl=10,
        filter_sql="satTye in [1, 5]",
        desc="satTye + satBin",
        agg_ops=['count', 'mean', 'std'],
        **params
    ) 
    feats = append_label(feats, df)
    plot_kde_grid(feats, hue=LABEL, n_col=3, common_norm=False)
    feats

    # %%
    """ 3. 不同类型的卫星个数"""
    feats, _ = aggregate_data(
        group_cols=['rid', 'satTye'], 
        attrs=['satIdentification'], 
        desc="satTye count",
        agg_ops=['nunique'], # 'count', 'unique'
        **params
    ) 

    feats = append_label(feats, df)
    plot_kde_grid(feats, hue=LABEL, n_col=3)
    feats
    # feats.fillna(0).astype(np.int8)

# %%
#! model
from sklearn.preprocessing import StandardScaler
from model.model import get_sklearn_model, plot_learning_curve, standardize_df


feats = df_converage.fillna(-1)

cols = list(feats.columns)
cols.remove(LABEL)

X, _, _ = standardize_df(feats, feats)
X = X[cols]
y = feats[LABEL]


# scaler = StandardScaler()

# X = pd.DataFrame(
#     scaler.fit_transform(feats[cols]),
#     columns=cols,
#     index=feats.index
# )


model_name = "KNN"
clf = get_sklearn_model(model_name)
plot_learning_curve(clf, model_name, X, y, train_sizes=[.05, .2, .4, .6, .8, 1.0]);

# %%
analyze_and_identify_correlated_columns(df_converage, plt_cfg={"annot": True})

# %%
