from pyspark.sql.types import DoubleType, StringType, FloatType
from pyspark.sql.functions import col, count, lag, lead, when, udf, collect_list, row_number, coalesce, lit, round
import builtins
from util.read_and_print_df import *
import util.read_and_print_df as rpdf
from pyspark.sql.window import Window
from functools import reduce
from pyspark.sql import Row
import math
import sys



def get_null_percentage_df_per_column(df):
    return df.select([(100 * count(when(col(c).isNull(), c))/count(lit(1))).alias(c) for c in df.columns])



def get_null_percentage_df_per_column_per_node(df):
    df_null = (df.groupBy('node')
                      .agg(*[(100 * count(when(col(c).isNull(), c))/count(lit(1))).alias(c + '%null') for c in df.columns]))
    cols_to_consider = [col(c) for c in df_null.columns if c not in ['node']]
    df_null = df_null.withColumn('sum', sum(c for c in cols_to_consider))
    df_null = df_null.withColumn('count', sum(c.isNotNull().cast('double') for c in cols_to_consider))
    df_null = df_null.withColumn('cols_mean%null', round(col('sum') / col('count'), 3))
    df_null = df_null.drop('sum', 'count')
    
    return df_null


def fill_and_drop_nan(df, spark=None):
    if spark:
        count_before = rpdf.speed_row_count(df, spark)
    else:
        count_before = df.count()

    # fill nvidia columns
    nvidia_cols = [c for c in df.columns if 'nvidia' in c]
    fill_values = {}
    for c in nvidia_cols:
        if df.schema[c].dataType == StringType():
            fill_values[c] = '{}'
        elif df.schema[c].dataType == DoubleType():
            fill_values[c] = 0.0
    df = df.fillna(fill_values, subset=[col(c) for c in nvidia_cols])
    
    # fill new columns in case prom test dataset is chosen
    if len(df.columns) > 300:
        new_cols = [c for c in df.columns[56:]]
        fill_values = {c: '{}' for c in new_cols}
        df = df.fillna(fill_values, subset=[col(c) for c in new_cols])

        fill_values = {}
        for c in new_cols:
            if df.schema[c].dataType == StringType():
                fill_values[c] = '{}'
            elif df.schema[c].dataType == DoubleType():
                fill_values[c] = 0.0
        df = df.fillna(fill_values, subset=[col(c) for c in new_cols])

    print('After NaN fill:')
    df.show(5, truncate=False)
    df = df.dropna()
    print('After NaN drop:')
    df.show(5, truncate=False)

    if spark:
        count_after = rpdf.speed_row_count(df, spark)
    else:
        count_after = df.count()
    print(f'NaN Removal dropped {builtins.round(100 * (count_before - count_after) / count_before, 2)}% of rows')
    return df



def fill_and_drop_nan_second_iteration(df):
    count_before = df.agg(count("*")).collect()[0][0]

    fill_values = {}
    for c in df.columns:
        if df.schema[c].dataType == StringType():
            fill_values[c] = '{}'
        elif df.schema[c].dataType == DoubleType():
            fill_values[c] = 0.0
    df = df.fillna(fill_values, subset=[col(c) for c in df.columns])

    print('After NaN fill:')
    df.show(5, truncate=False)
    df = df.dropna()
    print('After NaN drop:')
    df.show(5, truncate=False)
    print(f'NaN Removal dropped {builtins.round(100 * (count_before - df.agg(count("*")).collect()[0][0]) / count_before, 2)}% of rows')
    return df


def get_nan_rows_with_lead_and_lag_rows(df, window, show_df=False):

    expr = reduce(lambda x, y: x | y, [col(c).isNull() for c in df.columns])
    has_null_name = 'has_null'

    df = df.withColumn(has_null_name, expr)
    # get leading and lagging rows with non nan values for each nan row
    df = df.withColumn('is_lag', when(lag(has_null_name).over(window) & ~col(has_null_name), True).otherwise(False))
    df = df.withColumn('is_lead', when(lead(has_null_name).over(window) & ~col(has_null_name), True).otherwise(False))

    if show_df:
        print('has_null_lead_lag')
        df.show(15, truncate=False)
        
    df = df.filter(df[has_null_name] | df['is_lag'] | df['is_lead'])
    df = df.drop('is_lag', 'is_lead')

    return df

def linear_interpolate_nan_rows(df, cols_to_interpolate, window, default_val_dict, epsilon = 0.0):

    def linear_interpolate(i, vals, default_val=0.0):

        if not math.isnan(vals[i]):
            return vals[i]
        
        no_prev_non_nan = False
        no_next_non_nan = False

        # Search for previous non-nan index
        j = i - 1
        while j >= 0 and math.isnan(vals[j]):
            j -= 1
        if j < 0:
            no_prev_non_nan = True

        # Search for next non-nan index
        k = i + 1
        while k < len(vals) and math.isnan(vals[k]):
            k += 1
        if k >= len(vals):
            no_next_non_nan = True

        if no_prev_non_nan and no_next_non_nan:
            # use default value if everything is nan
            return default_val + epsilon
        elif no_prev_non_nan:
            # take next non-nan value assume a constant time series
            return vals[k] + epsilon
        elif no_next_non_nan:
            # take previous non-nan value assume a constant time series
            return vals[j] + epsilon

        # Compute interpolated value at index i
        x0 = j
        x1 = k
        y0 = vals[j]
        y1 = vals[k]
        xi = i
        yi = y0 + (xi - x0) * (y1 - y0) / (x1 - x0)
        
        # add epsilon to hint that this value is interpolated
        return yi + epsilon

    linear_interpolate_udf = udf(linear_interpolate, FloatType())
    df = df.withColumn('node_index', row_number().over(window) - 1)

    for c in cols_to_interpolate:
        if c in default_val_dict.keys():
            default_val = default_val_dict[c]
        elif 'default' in default_val_dict.keys():
                default_val = default_val_dict['default']
        else:
            default_val = 0.0
        df = df.withColumn(c, coalesce(col(c), lit(float("nan"))))
        df = df.withColumn(c , linear_interpolate_udf(col('node_index'), collect_list(c).over(window.rowsBetween(-sys.maxsize, sys.maxsize)), lit(default_val)))

    return df
