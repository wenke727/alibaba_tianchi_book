import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool


def add_agg_feature_names(df, df_group, group_cols, value_col, agg_ops, col_names):
    # df: 添加特征的dataframe
    # df_group: 特征生成的数据集
    # group_cols: group by 的列
    # value_col: 被统计的列
    # agg_ops:处理方式 包括：count,mean,sum,std,max,min,nunique
    # colname: 新特征的名称
    df_group[value_col] = df_group[value_col].astype("float")
    df_agg = pd.DataFrame(
        df_group.groupby(group_cols)[value_col].agg(agg_ops)
    ).reset_index()
    df_agg.columns = group_cols + col_names
    df = df.merge(df_agg, on=group_cols, how="left")
    return df

def add_agg_feature(df, df_group, group_cols, value_col, agg_ops, keyword):
    # 统计特征处理函数
    # 名称按照keyword+'_'+value_col+'_'+op 自动增加
    col_names = []
    for op in agg_ops:
        col_names.append(keyword + "_" + value_col + "_" + op)
    df = add_agg_feature_names(df, df_group, group_cols, value_col, agg_ops, col_names)
    return df

""" self """
import pandas as pd
import numpy as np
from multiprocessing import Pool

def parallel_aggregate(df_chunk):
    # df_chunk = df_chunk.apply(np.array)
    _df = pd.json_normalize(df_chunk.apply(lambda x: agg_funcs(np.array(x))))
    _df.index = df_chunk.index

    return _df

def npartition(df, n):
    chunk_size = len(df) // n + (len(df) % n > 0)
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]

def agg_funcs(x, funcs=['mean', 'std', '50%']):
    _dict = {
        'mean': np.mean(x),
        'std': np.std(x),
        'min': np.min(x),
        'max': np.max(x),
        '25%': np.quantile(x, 0.25),
        '50%': np.quantile(x, 0.5),
        '75%': np.quantile(x, 0.75)
    }
    
    return {i: _dict[i] for i in funcs }
   
def join_multi_columns(dfs):
    if len(dfs.columns.names) == 1:
        return dfs
    
    # modify columns
    columns = []
    for values in dfs.columns:
        lst = []
        for val, prefix in zip(values, dfs.columns.names):
            if prefix is None:
                lst.append(val)
            else:
                lst.append(f"{prefix}_{val}")
        columns.append("-".join(lst[::-1]))
    dfs.columns = columns
    
    return dfs    

def aggregate_data(data, 
                 group_keys, 
                 attrs, 
                 bin_att=None, 
                 intvl=25, 
                 filter_sql=None, 
                 n_partitions=4, 
                 feat_lst=[], 
                 tag_2_feats={}, 
                 desc=None):
    """
    Process the dataframe by grouping, binning, and aggregating.
    """
    if filter_sql:
        data = data.query(filter_sql)

    if bin_att and bin_att in data.columns:
        bin_col = bin_att[:5] + "Bin"
        data[bin_col] = (data[bin_att] // intvl).astype(np.int8)
        group_keys.append(bin_col)

    records = data.groupby(group_keys).agg({attr: list for attr in attrs})

    # Apply aggregation function
    df_chunks = list(npartition(records, n_partitions))
    
    lst = []
    for attr in attrs:
        if n_partitions == 1:
            records[attr] = records[attr].apply(np.array)
            _df = pd.json_normalize(records[attr].apply(agg_funcs))    
        else:
            _df_chunks = [df[attr] for df in df_chunks]
            pool = Pool(n_partitions)
            results = pool.map(parallel_aggregate, _df_chunks)
            _df = pd.concat(results)
        _df.index = records.index
        _df.columns = [f'{attr}_{i}' for i in _df.columns]
        lst.append(_df)
    dfs = pd.concat(lst, axis=1)

    for _ in group_keys[1:]:
        dfs = dfs.unstack()

    dfs = join_multi_columns(dfs)
    
    dfs.fillna(-1, inplace=True)
    feat_lst.append(dfs)
    if desc is None:
        desc = '-'.join(group_keys)
    tag_2_feats[desc] = dfs.columns.tolist()

    return dfs, records



# 你可以在此调用 process_data 函数
# 示例：result, records = process_data(data, group_cols, attrs, ...)
if __name__ == "__main__":
    dfs, df_processed = aggregate_data(df_sat, group_cols=['rid'], 
                                    attrs=['satDb', 'elevation'], 
                                    bin_att='satDb', 
                                    intvl=10, 
                                    filter_sql='satDb > 0 or isUsed == True'
                                    )


