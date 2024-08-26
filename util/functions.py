import psutil
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import DoubleType, StructType, StringType, MapType
from pyspark.sql.functions import col, count, when, isnan, lit, format_number, first, least, greatest, coalesce, from_json, expr, collect_set, round, countDistinct
from itertools import zip_longest
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from time import time
import builtins


def get_attributes_schema(df, spark):
    columns = ['name', 'dataType', 'null_percentage', 'example_value']
    null_percentage_df = df.select([(100 * count(when(col(c).isNull(), c))/count(lit(1))).alias(c) for c in df.columns]).collect()[0]

    def find_first_non_null(col_name):
        return first(col(col_name), ignorenulls=True).alias(col_name)

    data = df.agg(*[find_first_non_null(name) for name in df.columns])
    data_list = [(name, df.schema[name].dataType.simpleString(), str(null_percentage_df[name]), str(row[name])) for name in df.columns for row in data.collect()]

    df_schema = spark.createDataFrame(data_list, columns)
    return df_schema



def print_df(df, df_schema, n_th_row_select=1):
    if n_th_row_select > 1:
        x = 'nd' if n_th_row_select == 2 else ('rd' if n_th_row_select == 3 else 'th')
        print(f'Selected every {n_th_row_select}{x} row:')
    print('Size:', df.agg(count("*")).collect()[0][0], 'x', len(df.columns))

    print('Attributes:')
    df_schema.show(500, truncate=False)

    print('Example:')
    df.show(5, truncate=False)



def print_missing_values_heatmap(df):

    figsize = (20, 10)
    null_array = np.array(df.select([df[c].isNull().alias(c) for c in df.columns]).collect())
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(null_array, cmap='YlGnBu', ax=ax, center=0.5)

    num_cols = len(df.columns)
    tick_positions = [x+0.5 for x in range(num_cols)]
    tick_labels = df.columns
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick_positions))
    ax.set_xticklabels(tick_labels, rotation=45, horizontalalignment='right')
    if num_cols > 200:
        plt.xticks(fontsize=4)
    elif num_cols > 100:
        plt.xticks(fontsize=6)
    elif num_cols > 20:
        plt.xticks(fontsize=8)
    elif num_cols > 10:
        plt.xticks(fontsize=10)
    else:
        plt.xticks(fontsize=12)

    ax.set_title('Missing Values')



def fill_and_drop_nan(df):
    count_before = df.agg(count("*")).collect()[0][0]

    # fill nvidia columns
    nvidia_cols = [c for c in df.columns if 'nvidia' in c]
    if len(nvidia_cols) > 0:
        nvidia_cols = [c for c in df.columns if 'nvidia' in c]
        fill_values = {c: '{}' for c in nvidia_cols}
        df = df.fillna(fill_values, subset=[col(c) for c in nvidia_cols])
    
    # fill new columns in prom test dataset
    if len(df.columns) > 56:
        new_cols = [c for c in df.columns[56:]]
        fill_values = {c: '{}' for c in new_cols}
        df = df.fillna(fill_values, subset=[col(c) for c in new_cols])

    print('After NaN fill:')
    df.show(5, truncate=False)
    df = df.dropna()
    print('After NaN drop:')
    df.show(5, truncate=False)
    print(f'NaN Removal dropped {builtins.round(100 * (count_before - df.agg(count("*")).collect()[0][0]) / count_before, 2)}% of rows')
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



def read_df(path, spark, n_th_row_select=1, show_df_overview=False, show_nan_heatmap=False, handle_nan=False):
    df = spark.read.parquet(path)

    if n_th_row_select > 1:
        df = df = df.withColumn('index', monotonically_increasing_id())
        df = df.filter(df.index % n_th_row_select == 0)
        df = df.drop('index')

    df_schema = get_attributes_schema(df, spark)
    if show_df_overview:
        print_df(df, df_schema, n_th_row_select)
    if show_nan_heatmap:
        print_missing_values_heatmap(df)

    if handle_nan:
        df = fill_and_drop_nan(df)
        
    return df, df_schema



def filter_for_double_type_and_show_correlation(df, spark):
    figsize = (20,10)

    df_double = df.select([col(c) for c in df.columns if isinstance(df.schema[c].dataType, DoubleType)])
    if df_double.agg(count("*")).collect()[0][0] == 0 or df_double.columns == []:
        print('No DoubleType attributes or values found')
        return df_double
    df_double_schema = get_attributes_schema(df_double, spark)
    print_df(df_double, df_double_schema)
    
    assembler = VectorAssembler(inputCols=df_double.columns, outputCol='features')
    df_transformed = assembler.transform(df_double).select('features')
    matrix = Correlation.corr(df_transformed, 'features').head()
    corr_matrix = matrix[0].toArray()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, center=np.nanmean(corr_matrix), yticklabels=df_double.columns, annot=True, ax=ax, cmap='coolwarm', fmt='.1f')

    num_cols = len(df_double.columns)
    tick_positions = [x+0.5 for x in range(num_cols)]
    tick_labels = df_double.columns
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick_positions))
    ax.set_xticklabels(tick_labels, rotation=45, horizontalalignment='right')
    if num_cols > 200:
        plt.xticks(fontsize=4)
    elif num_cols > 100:
        plt.xticks(fontsize=6)
    elif num_cols > 20:
        plt.xticks(fontsize=8)
    elif num_cols > 10:
        plt.xticks(fontsize=10)
    else:
        plt.xticks(fontsize=12)

    ax.set_title('Data Correlation')

    return df_double



def extract_json(df, column_name, show_df=False):
    print(f'Extracting {column_name} json')
    json_schema = MapType(StringType(), StringType())
    df = df.withColumn(column_name, from_json(col(column_name), json_schema))

    distinct_keys = df.selectExpr(f'explode(map_keys({column_name})) as key') \
                  .agg(expr('collect_set(key)').alias('distinct_keys')) \
                  .collect()[0]['distinct_keys']
    keys = list(distinct_keys)
    
    print(f'Extracted {len(keys)} keys')
    print(keys)

    json_schema = StructType().add('key', StringType(), True).add('value', StringType(), True)

    for key in keys:
        df = df.withColumn(f'{column_name}-{key}', col(column_name)[key].cast(DoubleType()))

    if show_df:
        df.show(5, truncate=False)
    return df



def aggregate_json(df, column_name, aggregate_values, show_df=False):   
    column_names = [c for c in df.columns if c.startswith(f'{column_name}-')]
    cols_to_consider = [col(c) for c in column_names]

    if len(cols_to_consider) == 1:
        if 'min' in aggregate_values:
            df = df.withColumn(f'{column_name}-min', round(cols_to_consider[0], 2).cast('double'))
        if 'mean' in aggregate_values or 'sum' in aggregate_values:
            df = df.withColumn(f'{column_name}-sum', round(cols_to_consider[0], 2).cast('double'))
            if 'mean' in aggregate_values:
                df = df.withColumn(f'{column_name}-non_null_count', sum(c.isNotNull().cast('double') for c in cols_to_consider).cast('double'))
                df = df.withColumn(f'{column_name}-mean', round(when(col(f'{column_name}-non_null_count') != 0, col(f'{column_name}-sum') / col(f'{column_name}-non_null_count')).otherwise(lit(None)), 2).cast('double'))
        if 'max' in aggregate_values:
            df = df.withColumn(f'{column_name}-max', round(cols_to_consider[0], 2).cast('double'))

    elif len(cols_to_consider) > 1:
        if 'min' in aggregate_values:
            df = df.withColumn(f'{column_name}-min', round(least(*cols_to_consider), 2).cast('double'))
        if 'mean' in aggregate_values or 'sum' in aggregate_values:
            df = df.withColumn(f'{column_name}-sum', round(sum(coalesce(c, lit(0.0)) for c in cols_to_consider), 2).cast('double'))
            if 'mean' in aggregate_values:
                df = df.withColumn(f'{column_name}-non_null_count', sum(c.isNotNull().cast('double') for c in cols_to_consider).cast('double'))
                df = df.withColumn(f'{column_name}-mean', round(when(col(f'{column_name}-non_null_count') != 0, col(f'{column_name}-sum') / col(f'{column_name}-non_null_count')).otherwise(lit(None)), 2).cast('double'))
        if 'max' in aggregate_values:
            df = df.withColumn(f'{column_name}-max', round(greatest(*cols_to_consider), 2).cast('double'))
    
    print("New values:")
    if show_df:
        df.show(5, truncate=False)

    df = df.drop(column_name, f'{column_name}-non_null_count', *column_names)
    if not 'sum' in aggregate_values:
        df= df.drop(f'{column_name}-sum')

    return df