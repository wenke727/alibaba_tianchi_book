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
from sklearn.metrics import classification_report

from feature import get_sat_coverage_dataframe
from vis import plot_satellite_distribution_seaborn
from model.vis import visualize_model_confusion_matrix
from model.preprocess import reduce_mem_usage
from model.featureEng import aggregate_data
from model.preprocess import analyze_and_identify_correlated_columns
from model.data_analysis_visualization import add_predictions_and_labels, visualize_confusion_features, plot_kde_grid
from model.featureSelection import null_importance_feature_selection, calculate_feature_importance_xgb
from model.model import get_sklearn_model, plot_learning_curve, standardize_df
from model.featureSelection import rfecv_feature_selection
from model.metric import flatten_classification_report
from model.dataframe import query_dataframe
from model.utils.serialization import load_checkpoint, save_checkpoint

import warnings
warnings.filterwarnings('ignore')

from cfg import DATA_FOLDER, LABEL, KEYWORD_2_COLUMN

FEAT_CKPT = "../data/gnss_feats.pkl"


""" 工具函数 """
def load_feats(fn=FEAT_CKPT):
    data = load_checkpoint(fn)
    df = data['df']
    df_sat = data['df_sat']
    feats = data['feats']
    feat_lst = data['feat_lst']
    tag_2_feats = data['tag_2_feats']

    return df, df_sat, feats, feat_lst, tag_2_feats

def save_feats(df, ad_sat, feats, feat_lst, tag_2_feats, fn=FEAT_CKPT):
    data = {
        "df": df,
        'df_sat': df_sat,
        "feats": feats, 
        "feat_lst": feat_lst, 
        "tag_2_feats": tag_2_feats, 
    }
    save_checkpoint(data, FEAT_CKPT)

def detect_outlier(bad_cases, df):
    fn_2_num = df.groupby('fn')['label'].count()
    fn_2_num.name = 'total_num'

    _df_stat = bad_cases.groupby('fn')[['aTime']].count().rename(columns={'aTime': 'num'})
    _df_stat = _df_stat.merge(fn_2_num, left_index=True, right_index=True)
    _df_stat.loc[:, 'ratio'] = _df_stat.num / _df_stat.total_num
    _df_stat.sort_values('ratio', ascending=False)
    
    return _df_stat

def get_sat_records(df, idxs, keep_atts=['fn', 'label', 'aTime', 'satCount', 'useSatCount']):
    _df = df.loc[idxs][keep_atts]

    sql = ['not (azimuth == 0 and elevation == 0)', 'isUsed == True', 'satDb > 0']
    sql =  ["rid in @idxs"] + sql
    sats = df_sat.query(' and '.join(sql)).groupby('rid').agg(list)

    db_lst = sats.satDb.apply(lambda x: np.sort(x).astype(int)[::-1])
    db_mean = db_lst.apply(np.mean)
    db_75 = db_lst.apply(lambda x: np.quantile(x, .75))
    _num = db_lst.apply(len)

    _df.loc[:, 'num'] = _num.fillna(0).astype(int)
    _df.loc[:, 'Db_mean'] = db_mean
    _df.loc[:, 'Db_75'] = db_75
    _df.loc[:, 'db_lst'] = db_lst
    
    return _df

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
    groups = pd.Series(groups)
    for train_idx, valid_idx in gkf.split(X, y, groups):
        # Yield the subsets of the data for the current split
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        logger.debug(f"Training: {list(np.unique(groups.iloc[train_idx]))}")
        logger.debug(f"Validation: {list(np.unique(groups.iloc[valid_idx]))}")
    
        if LABEL in list(X_train):
            X_train.drop(columns=LABEL, inplace=True)
            X_valid.drop(columns=LABEL, inplace=True)
 
        yield X_train, y_train, X_valid, y_valid

def append_label(feats, df):
    return feats.merge(df[[LABEL]], left_index=True, right_index=True)

""" 读取函数 """
def convert_timestamp_to_datetime_custom_unit(df: pd.DataFrame, column_name: str, unit: str = 'ms'):
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

def explode_sat_data(df: pd.DataFrame):
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

def extract_data_from_zip(zip_filename: str, keyword: str):
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

def extract_data_from_directory(folder: str, keywords: str, unit: str = 'ms'):
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

def feature_pipline_for_GNSS(df_sat, df, base_cols=['satCount', 'useSatCount'], n_jobs=1):
    tag_2_feats = {'base': base_cols}
    feat_lst = [df[base_cols]]
    params = {
        'data': df_sat, 
        'feat_lst': feat_lst,
        'tag_2_feats': tag_2_feats, 
        'n_partitions': n_jobs,
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
    # feats = append_label(feats, df)
    # plot_kde_grid(feats, hue=LABEL, n_col=3);

    """ 3. 卫星覆盖程度"""
    df_converage = get_sat_coverage_dataframe(df_sat, ['rid'])
    feat_lst.append(df_converage)
    tag_2_feats['converage'] = list(df_converage)
    # plot_kde_grid(df_converage, hue=LABEL, n_col=3)
    # sns.pairplot(data=df_converage, hue=LABEL)

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
    # feats = append_label(feats, df)
    # plot_kde_grid(feats, hue=LABEL, n_col=5, common_norm=False);

    # 1.3 GNSS Type
    feats, _ = aggregate_data(
        group_cols=['rid', 'satTye'], 
        attrs=['satDb'], 
        filter_sql="satTye in [1, 5]",
        desc="satTye",
        agg_ops=['count', 'mean', 'std', '50%', '75%'],
        **params
    ) 
    # feats = append_label(feats, df)
    # plot_kde_grid(feats, hue=LABEL, n_col=5, common_norm=False);

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
    # feats = append_label(feats, df)
    # plot_kde_grid(feats, hue=LABEL, n_col=3, common_norm=False)

    """ 2. 不同类型的卫星个数"""
    feats, _ = aggregate_data(
        group_cols=['rid', 'satTye'], 
        attrs=['satIdentification'], 
        desc="satTye count",
        agg_ops=['nunique'], # 'count', 'unique'
        **params
    ) 

    # feats = append_label(feats, df)
    # plot_kde_grid(feats, hue=LABEL, n_col=3)

    info = "\n".join([f"\t'{k}': {v}" for k, v in tag_2_feats.items()] + ['}'])
    logger.debug("feature_columns = {\n" + f"{info}")
    
    return feat_lst, tag_2_feats

def postprocess_feature(df, feat_lst, tag_2_feats, drop_high_coor_col_thred=0.98):
    df_feats = pd.concat(feat_lst[:], axis=1).fillna(0)
    df_feats = append_label(df_feats, df)

    # 共线分析
    if drop_high_coor_col_thred is not None and drop_high_coor_col_thred > 0:
        columns_to_drop = analyze_and_identify_correlated_columns(
            df_feats, 
            threshold=drop_high_coor_col_thred, 
            plt_cfg={"annot": True}
    )
    df_feats.drop(columns=columns_to_drop, inplace=True)
    
    for k, vals in tag_2_feats.items():
        for val in vals:
            if val in columns_to_drop:
                vals.remove(val)

    return df_feats, tag_2_feats


#%%
if __name__ == "__main__":
    df = extract_data_from_directory(DATA_FOLDER, ["GNSS", 'light', 'mobile'])
    df_sat = explode_sat_data(df)

    # sat = df_sat.query('rid==99')
    # plot_satellite_distribution_seaborn(sat)

    # step 1: delete `NaN` 
    df_sat = reduce_mem_usage(df_sat)
    df_sat.satFrequency = np.round((df_sat.satFrequency / 1e6).values, 2)

    # step 2: delete `cache`
    df_sat = delete_cache_gnss_records(df_sat)

    # step 3: Pipeline 
    feat_lst, tag_2_feats = feature_pipline_for_GNSS(df_sat, df)
    feats, tag_2_feats = postprocess_feature(df, feat_lst, tag_2_feats, drop_high_coor_col_thred=0.99)

    # save
    save_feats(df, df_sat, feats, feat_lst, tag_2_feats, fn=FEAT_CKPT)


#%%
df, df_sat, feats, feat_lst, tag_2_feats = load_feats(FEAT_CKPT)

""" step 4: 切分数据 """
X_train, y_train, X_valid, y_valid = next(group_kfold_split(feats, feats[LABEL], df.loc[feats.index].fn, n_splits=5))

#%% 
""" step 5：特征选择 """
# feature_comparison, useful_feature_names = null_importance_feature_selection(X_train, y_train, X_train.columns, 200, 42)
useful_feature_names = rfecv_feature_selection(X_train, y_train, get_sklearn_model("LR"))

#%%
importance_df, unimportance_df = calculate_feature_importance_xgb(
    X_train[useful_feature_names], y_train, num_boost_round=200, importance_type='cover')
importance_df

# %%
""" step 6. model """
# cols = X_train.columns
cols = useful_feature_names
X_train_scaled, X_valid_scaled, scaler = standardize_df(X_train[cols], X_valid[cols])


model_name = "LR"
clf = get_sklearn_model(model_name)
clf.fit(X_train_scaled, y_train)

plot_learning_curve(clf, model_name, X_train_scaled, y_train, train_sizes=[.05, .2, .4, .6, .8, 1.0]);

# %%
cfg = {
    'labels': [False, True],
    'target_names': ['Out', "In"],
}

visualize_model_confusion_matrix(clf, X_train_scaled, y_train, ['out', 'in'])
metric = classification_report(y_train, clf.predict(X_train_scaled), **cfg)
logger.debug(f"Training:\n{metric}")

flatten_classification_report(y_train, clf.predict(X_train_scaled), **cfg)

#%%
visualize_model_confusion_matrix(clf, X_valid_scaled, y_valid, ['out', 'in'])
metric = classification_report(y_valid, clf.predict(X_valid_scaled), **cfg)
logger.warning(f"Validation:\n{metric}")

# %%
importance_df, unimportance_df = calculate_feature_importance_xgb(
    X_train_scaled[useful_feature_names], y_train, num_boost_round=200, importance_type='cover')
ordered_cols = importance_df.Feature.values.tolist()
ordered_cols

#%%
X_with_labels = add_predictions_and_labels(clf, X_train_scaled, y_train)

#%%
analyzed_data = visualize_confusion_features(
    X_with_labels, scaler=scaler, suptitle="Confusion: Out -> In", gt=False, pred=True, cols=ordered_cols)
idxs = analyzed_data[2].index
df.loc[idxs]

#%%
get_sat_records(df, idxs, keep_atts=['fn', 'aTime', 'satCount', 'useSatCount'])


#%%
analyzed_data = visualize_confusion_features(
    X_with_labels, scaler=scaler, suptitle="Confusion: In -> Out", gt=True, pred=False, cols=ordered_cols)
idxs = analyzed_data[2].index
#%%

get_sat_records(df, idxs, keep_atts=['fn', 'aTime', 'satCount', 'useSatCount'])

# %%
detect_outlier(_df, df)


# %%
