from pyspark.sql.types import DoubleType, StructType, StringType, MapType
from pyspark.sql.functions import col, when, lit, least, greatest, coalesce, from_json, expr, round



def extract_json(df, column_name, show_df=False, casting_type=DoubleType()):

    json_schema = MapType(StringType(), StringType())
    # convert json string to dict
    df = df.withColumn(column_name, from_json(col(column_name), json_schema))
    # get all keys from dict
    keys = extract_keys(df, column_name)

    # make every key a new column
    for key in keys:
        df = df.withColumn(f'{column_name}-{key}', col(column_name)[key].cast(casting_type))

    if show_df:
        df.show(5, truncate=False)
    return df

def extract_keys(df, column_name):
    print(f'Extracting {column_name} json')

    distinct_keys = df.selectExpr(f'explode(map_keys({column_name})) as key') \
                  .agg(expr('collect_set(key)').alias('distinct_keys')) \
                  .collect()[0]['distinct_keys']
    keys = list(distinct_keys)
    
    print(f'Extracted {len(keys)} keys')
    print(keys)

    return keys



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
