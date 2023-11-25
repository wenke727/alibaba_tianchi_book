import pandas as pd
import numpy as np


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
