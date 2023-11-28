import pandas as pd
import numpy as np
from loguru import logger


def longest_consecutive(nums, length=12, verbose=False):
    if not nums:
        if verbose:
            logger.debug(f"LIS: 0, nums: {nums}")
        return 0

    nums = sorted(set(nums))
    if len(nums) == length:
        if verbose:
            logger.debug(f"LIS: {length}, nums: {nums}")
        return length

    nums_extended = nums + [i + length for i in nums]
    dp = [1] * len(nums_extended)
    max_len = 1

    for i in range(1, len(nums_extended)):
        if nums_extended[i] - nums_extended[i - 1] == 1:
            dp[i] = dp[i - 1] + 1
        max_len = max(max_len, dp[i])

    if verbose:
        logger.debug(f"LIS: {max_len:2d}, nums: {nums}")

    return max_len

def satellite_coverage_metric(df, elevation_interval=30, azimuth_interval=30):
    elevation_bin = (df['elevation'] // elevation_interval).astype(int)
    df['elevation_bin'] = elevation_bin * elevation_interval

    azimuth_bin = (df['azimuth'] // azimuth_interval).astype(int)
    df['azimuth_bin'] = azimuth_bin * azimuth_interval
    
    coverage_df = df.groupby(['elevation_bin', 'azimuth_bin']).agg({
        'satDb': ['count', 'mean']
    }).reset_index()
    
    coverage_df.columns = ['elevation_bin', 'azimuth_bin', 'sat_count', 'snr_mean']
    
    total_bins = (360 // azimuth_interval) * (90 // elevation_interval)
    occupied_bins = coverage_df['sat_count'].gt(0).sum()
    coverage_percentage = (occupied_bins / total_bins)
    average_snr = coverage_df['snr_mean'].mean()
    snr_std = coverage_df['snr_mean'].std()
    
    mask = elevation_bin == 0
    lis_0 = longest_consecutive(azimuth_bin[mask].tolist())
    mask = elevation_bin == 1
    lis_1 = longest_consecutive(azimuth_bin[mask].tolist())
    mask = elevation_bin == 2
    lis_2 = longest_consecutive(azimuth_bin[mask].tolist())
    lis = longest_consecutive(np.unique(azimuth_bin).tolist())

    summary = {
        # 'total_bins': total_bins,
        # 'occupied_bins': occupied_bins,
        'elevation_bin': elevation_bin.values,
        'azimuth_bin': azimuth_bin.values,
        'coverage_percentage': coverage_percentage,
        'average_snr': average_snr,
        'snr_std_dev': snr_std,
        'lis': lis,
        'lis_0': lis_0,
        'lis_1': lis_1,
        'lis_2': lis_2,
    }
    
    return summary

def get_sat_coverage_dataframe(df_sat, group_cols=['rid'], bin_att=False):
    data = df_sat.groupby(group_cols).apply(satellite_coverage_metric)

    df_coverage = pd.json_normalize(data=data)
    df_coverage.index = data.index
    
    if not bin_att:
        df_coverage.drop(columns=['elevation_bin', 'azimuth_bin'], inplace=True)
    
    return df_coverage
