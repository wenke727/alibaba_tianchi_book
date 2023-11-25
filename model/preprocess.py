import pandas as pd
import numpy as np
from loguru import logger


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

