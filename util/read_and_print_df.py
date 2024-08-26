#########################################################################################
# Set your paths to the datasets here before running any script                         #
#########################################################################################

path_job_dataset = 'change to your dataset dir path/slurm_table_cleaned.parquet'
path_node_dataset = 'change to your dataset dir path/prom_table_cleaned.parquet'
path_job_node_joined_dataset = 'change to your dataset dir path/prom_slurm_joined.parquet'
path_node_hardware_info = 'change to your dataset dir path/node_hardware_info.parquet'



from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StructType, StructField, StringType
from pyspark.sql.functions import col, count, when, lit, first, avg, stddev, round
import pyspark.sql.functions as F
from pyspark.sql import Row
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from util.handle_nan import *
import builtins
import os
import psutil
import shutil



def get_spark_session():
    mem_fraction_gb = int(psutil.virtual_memory().total * 10**(-9) / 2)
    core_count = os.cpu_count()
    if core_count == None:
        use_cores = 1
    else:
        use_cores = builtins.max(core_count - 2, 2)
    
    print(f"Assigning {mem_fraction_gb} GB of memory per spark driver and executor, and use {use_cores} cores.")

    spark = SparkSession.builder \
        .appName('Read Parquet File') \
        .master(f'local[{use_cores}]') \
        .config('spark.driver.memory', f'{mem_fraction_gb}g') \
        .config('spark.executor.memory', f'{mem_fraction_gb}g') \
        .config('spark.driver.maxResultSize', f'{mem_fraction_gb}g') \
        .config('spark.driver.extraJavaOptions', '-Xss1g') \
        .config('spark.sql.codegen.wholeStage', 'false') \
        .getOrCreate()
    return spark


def get_attributes_schema(df, spark):
    columns = ['name', 'dataType', 'null_percentage', 'example_value', 'mean', 'std', 'coeff_var']
    schema_list = []
    for c in columns:
        schema_list.append(StructField(c, StringType(), True))
    schema = StructType(schema_list)


    null_percentage_list = get_null_percentage_df_per_column(df).collect()[0]

    def find_first_non_null(col_name):
        return first(col(col_name), ignorenulls=True).cast('string').alias(col_name)

    examples = df.agg(*[find_first_non_null(name) for name in df.columns])
    mean_list = [df.agg(round(avg(when(col(c).isNotNull(), col(c))).alias(c), 2)).collect()[0][0] if isinstance(df.schema[c].dataType, DoubleType) else None for c in df.columns]
    std_list = [df.agg(round(stddev(when(col(c).isNotNull(), col(c))).alias(c), 2)).collect()[0][0] if isinstance(df.schema[c].dataType, DoubleType) else None for c in df.columns]
    coeff_var_list = [builtins.round(sd / me, 4) if sd != None and me != None and me > 0.0 else None for (sd, me) in zip(std_list, mean_list)]
    data_tuples_list = [(name, df.schema[name].dataType.simpleString(), str(null_percentage_list[name]), str(row[name]), mean_list[i], std_list[i], coeff_var_list[i]) for (i, name) in enumerate(df.columns) for row in examples.collect()]

    df_schema = spark.createDataFrame(data_tuples_list, schema=schema)
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



def print_missing_values_heatmap_paper(df):

    figsize = (20, 10)
    null_array = np.array(df.select([df[c].isNull().alias(c) for c in df.columns]).collect())
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = sns.color_palette("Reds", 5).as_hex()
    
    sns.heatmap(null_array, cmap=cmap, ax=ax, center=0.5)

    num_cols = len(df.columns)
    tick_positions = [x+0.5 for x in range(num_cols)]
    tick_labels = df.columns
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick_positions))
    ax.set_xticklabels(tick_labels, rotation=45, horizontalalignment='right')

    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    num_rows = df.agg(count("*")).collect()[0][0]
    tick_labels = range(0, num_rows, 25000)
    
    ax.set_yticks(tick_labels)
    tick_labels = [f'{x:,}' for x in tick_labels]
    ax.set_yticklabels(tick_labels, horizontalalignment='right')

    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([.0, .20, .40, .60, .80, 1.0])
    colorbar.set_ticklabels(['0\%', '20\%', '40\%', '60\%', '80\%', '100\%'])

    plt.yticks(fontsize=20)
    plt.title('Missing Values', fontsize=20)


def read_df(path, spark, n_th_row_select=1, show_df_overview=False, show_nan_heatmap=False, handle_nan=False, sort=False):
    df = spark.read.parquet(path)

    if sort:
        df = df.orderBy([col('node'), col('timestamp')])

    if n_th_row_select > 1:
        df = df.withColumn('index', monotonically_increasing_id())
        df = df.filter(df.index % n_th_row_select == 0)
        df = df.drop('index')

    if sort:
        df = df.orderBy([col('node'), col('timestamp')])


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
    for method in ['pearson', 'spearman']:
        matrix = Correlation.corr(df_transformed, 'features', method).head()
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

        ax.set_title(f'{method.capitalize()} Correlation')
        plt.show()

    return df_double

def speed_row_count(df, spark):
    first_column = df.select(df.columns[0])
    path = "temp_file.parquet"
    first_column.write.parquet(path, mode='overwrite')
    temp_df = spark.read.parquet(path)
    count = temp_df.count()
    shutil.rmtree(path)
    return count

def checkpoint_df(df, spark, checkpoint_path):
    df.write.parquet(checkpoint_path, mode='overwrite')
    return spark.read.parquet(checkpoint_path)

def get_dummy_df(spark):
    data = [
        Row(id=1, timestamp='2022-08-01 23:00:00', node='A', node_power_usage=None),
        Row(id=2, timestamp='2022-08-01 23:00:30', node='A', node_power_usage=30.0),
        Row(id=3, timestamp='2022-08-01 23:01:00', node='A', node_power_usage=None),
        Row(id=4, timestamp='2022-08-01 23:01:30', node='A', node_power_usage=None),
        Row(id=5, timestamp='2022-08-01 23:02:00', node='A', node_power_usage=None),
        Row(id=6, timestamp='2022-08-01 23:02:30', node='A', node_power_usage=30.0),
        Row(id=7, timestamp='2022-08-01 23:03:00', node='A', node_power_usage=40.0),
        Row(id=8, timestamp='2022-08-01 23:03:30', node='A', node_power_usage=45.0),
        Row(id=9, timestamp='2022-08-01 23:04:00', node='A', node_power_usage=50.0),
        Row(id=10, timestamp='2022-08-01 23:04:30', node='A', node_power_usage=None),
        Row(id=11, timestamp='2022-08-01 23:00:00', node='B', node_power_usage=45.0),
        Row(id=12, timestamp='2022-08-01 23:00:30', node='B', node_power_usage=50.0),
        Row(id=13, timestamp='2022-08-01 23:01:00', node='B', node_power_usage=None),
        Row(id=14, timestamp='2022-08-01 23:01:30', node='B', node_power_usage=80.0),
        Row(id=15, timestamp='2022-08-01 23:02:00', node='B', node_power_usage=85.0)
    ]

    df = spark.createDataFrame(data)
    return df

gpu_nodes = {
    "r28n1", "r28n2", "r28n3", "r28n4", "r28n5",
    "r29n1", "r29n2", "r29n3", "r29n4", "r29n5",
    "r30n1", "r30n2", "r30n3", "r30n4", "r30n5", "r30n6", "r30n7",
    "r31n1", "r31n2", "r31n3", "r31n4", "r31n5", "r31n6",
    "r32n1", "r32n2", "r32n3", "r32n4", "r32n5", "r32n6", "r32n7",
    "r33n2", "r33n3", "r33n5", "r33n6",
    "r34n1", "r34n2", "r34n3", "r34n4", "r34n5", "r34n6", "r34n7",
    "r35n1", "r35n2", "r35n3", "r35n4", "r35n5",
    "r36n1", "r36n2", "r36n3", "r36n4", "r36n5",
    "r38n1", "r38n2", "r38n3", "r38n4", "r38n5",
}


def extract_nodes(c):
    
    def process_rack(g):
        # data for a rack is in the r13n[1,2,3] or r13n1 format
        # The r is lost during a previous split
        # We use the identify and extract rack and node identifiers
        rack_id = F.regexp_extract(g, "([0-9]+)n", 1)
        node_ids = F.regexp_extract(g, "n\[?([0-9,]+)", 1)
        node_id_list = F.split(node_ids, ",")
        combined_ids = F.transform(node_id_list, lambda nid: F.concat(lit("r"), rack_id, lit("n"), nid))
        return combined_ids
        
    splits = F.split(c, ",r")
    all_racks = F.transform(splits, lambda x: process_rack(x))
    return F.flatten(all_racks)



# expects pyspark df
# Extracts the nodes that are gpu nodes in Prometheus data, or, in case of SLURM dataset, ML jobs run on gpu nodes
# 1 indicates that the node is a gpu node (job is ML job), 0 otherwise
def get_gpu_node_col(df, node_col_name):
    df = df.withColumn("nodez", extract_nodes(col(node_col_name)))
    df = df.withColumn("gpu_node", F.array_intersect(F.array(*[lit(x) for x in gpu_nodes]), col("nodez")))
    df = df.drop("nodez")
    df = df.withColumn("gpu_node", F.when(F.size(col("gpu_node")) > 0, 1).otherwise(0))
    return df



# expects pandas df
# Creates a column that indicates if a job is ML (was executed on GPU nodes)
def mark_ml_df(df):
    split_nodes_ls = []
    for s in df['node']:
        split_nodes_ls.append(split_nodes(s))
    df["split_nodes"] = split_nodes_ls
    node_type_l = []
    for i in df["split_nodes"]:
        if any(n in gpu_nodes for n in i):
            node_type = 1 # 1: ml jobs
        else: 
            node_type = 0 # 0: generic jobs
        node_type_l.append(node_type)
    df["is_ml"] = node_type_l

# parses 'node' strings like r12n[1-30,32] to r12n1, r12n2 ... r12n30, r12n32  
def split_nodes(s):
    if s is None or len(s) == 0:
        return set()
    
    s = s.replace("\r\n", "").replace("\n", "").replace("\t", "")

    start = 0
    index = 0
    rack_chunks = []
    in_bracket = False
    while index < len(s):  # Separate them in parts like r12n[1-30,32] or r13n1
        if s[index] == "[":
            in_bracket = True
        elif s[index] == "]":
            in_bracket = False
        elif s[index] == "," and not in_bracket:
            rack_chunks.append(s[start: index])
            start = index + 1
        index += 1
    rack_chunks.append(s[start: index])  # Add the last line
    
    node_names = set()

    for rack_chunk in rack_chunks:
        if "[" in rack_chunk:
            prefix, postfix = rack_chunk.split("[")
            postfix = postfix[:-1]  # Remove the last bracket
            nodes = postfix.split(",")
            for node in nodes:
                if "-" in node:
                    start, end = node.split("-")
                    if not start.isnumeric() or not end.isnumeric():
                        continue
                    for i in range(int(start), int(end) + 1):
                        node_names.add("{}{}".format(prefix, i))
                else:
                    node_names.add("{}{}".format(prefix, node))
        else:
            node_names.add(rack_chunk)

    return node_names
