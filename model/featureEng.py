import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial


def npartition(df, n):
    chunk_size = len(df) // n + (len(df) % n > 0)
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size]

def agg_funcs(x, funcs=["mean", "std", "50%"]):
    _dict = {
        "count": len(x),
        "unique": np.unique(x),
        "nunique": len(np.unique(x)),
        "mean": np.mean(x),
        "std": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "25%": np.quantile(x, 0.25),
        "50%": np.quantile(x, 0.5),
        "75%": np.quantile(x, 0.75),
    }

    return {i: _dict[i] for i in funcs if i in _dict.keys()}

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

def parallel_aggregate(df_chunk, agg_ops):
    # df_chunk = df_chunk.apply(np.array)
    _df = pd.json_normalize(df_chunk.apply(lambda x: agg_funcs(np.array(x), agg_ops)))
    _df.index = df_chunk.index

    return _df

def aggregate_data(
    data,
    group_cols,
    attrs,
    bin_att=None,
    intvl=25,
    max_bin=4,
    filter_sql=None,
    n_partitions=4,
    agg_ops=["mean", "std", "50%"],
    feat_lst=[],
    tag_2_feats={},
    desc=None,
):
    """
    Process the dataframe by grouping, binning, and aggregating.
    """
    data = data.copy()
    if filter_sql:
        data = data.query(filter_sql)

    if bin_att and bin_att in data.columns:
        bin_col = bin_att[:5] + "Bin"
        data.loc[:, bin_col] = (data[bin_att] // intvl).astype(np.int8)
        data.loc[data[bin_col] >= max_bin, bin_col] = max_bin
        group_cols.append(bin_col)

    records = data.groupby(group_cols).agg({attr: list for attr in attrs})

    # Apply aggregation function
    df_chunks = list(npartition(records, n_partitions))

    lst = []
    for attr in attrs:
        if n_partitions == 1:
            records[attr] = records[attr].apply(np.array)
            _df = pd.json_normalize(
                records[attr].apply(lambda x: agg_funcs(x, funcs=agg_ops))
            )
        else:
            _df_chunks = [df[attr] for df in df_chunks]
            pool = Pool(n_partitions)
            _parallel_aggregate = partial(parallel_aggregate, agg_ops=agg_ops)
            results = pool.map(_parallel_aggregate, _df_chunks)
            _df = pd.concat(results)
        _df.index = records.index
        _df.columns = [f"{attr}_{i}" for i in _df.columns]
        lst.append(_df)
    dfs = pd.concat(lst, axis=1)

    for _ in group_cols[1:]:
        dfs = dfs.unstack()

    dfs = join_multi_columns(dfs)

    # dfs.fillna(-1, inplace=True)
    feat_lst.append(dfs)
    if desc is None:
        desc = "-".join(group_cols)
    tag_2_feats[desc] = dfs.columns.tolist()

    return dfs, records


""" Special """
def longestConsecutive(nums, max_val=11):
    if not nums:
        return 0

    # 标准的DP方法来找最长连续子序列
    def dp_longest_consecutive(nums):
        nums = sorted(set(nums))  # 去除重复并排序
        dp = [1] * len(nums)
        max_len = 0 if not nums else 1

        for i in range(1, len(nums)):
            if nums[i] - nums[i - 1] == 1:
                dp[i] = dp[i - 1] + 1
            max_len = max(max_len, dp[i])

        return max_len

    # 调整数组，将0视为 max_val + 1
    nums_adjusted = [num if num != 0 else max_val + 1 for num in nums]
    nums_adjusted += [num + max_val + 1 for num in nums if num < max_val]

    # 对原始数组和调整后的数组应用DP
    max_len_original = dp_longest_consecutive(nums)
    max_len_adjusted = dp_longest_consecutive(nums_adjusted)

    return max(max_len_original, max_len_adjusted)


# 你可以在此调用 process_data 函数
# 示例：result, records = process_data(data, group_cols, attrs, ...)
if __name__ == "__main__":
    dfs, df_processed = aggregate_data(
        df_sat,
        group_cols=["rid"],
        attrs=["satDb", "elevation"],
        bin_att="satDb",
        intvl=10,
        filter_sql="satDb > 0 or isUsed == True",
    )
