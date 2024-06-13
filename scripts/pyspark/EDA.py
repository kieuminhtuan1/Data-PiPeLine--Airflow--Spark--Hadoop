from pyspark.sql import SparkSession
import logging
import matplotlib.pyplot as plt

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Credit Risk EDA") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Path to data
data_path = "hdfs://namenode:9000/home_credit/data/"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to read and display Parquet


def read_and_show_parquet(file_name):
    df = spark.read.parquet(data_path + file_name)
    logging.info(df.show())
    return df


# Read train and test data from Parquet
data_app_train = read_and_show_parquet("application_train.parquet")


# Display schema
logging.info("Schema:")
data_app_train.printSchema()

# Summary statistics
logging.info("Summary statistics:")
summary = data_app_train.describe()
summary.show()

# Count of label classes
logging.info("Label distribution:")
label_counts = data_app_train.groupBy("TARGET").count()
label_counts.show()

# Show some sample data
logging.info("Sample data:")
data_app_train.show(5)

# Convert label_counts to pandas for plotting
label_counts_pd = label_counts.toPandas()

# Plot label distribution
plt.figure(figsize=(8, 6))
plt.bar(label_counts_pd['TARGET'], label_counts_pd['count'], color='skyblue')
plt.xlabel('TARGET')
plt.ylabel('Count')
plt.title('Distribution of TARGET')
plt.savefig('/opt/target_distribution.png')

# Exploratory plots for some numeric features
numeric_columns = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH']
for column in numeric_columns:
    column_pd = data_app_train.select(column).toPandas()

    plt.figure(figsize=(8, 6))
    plt.hist(column_pd[column], bins=50, color='skyblue')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Distribution of {column}')
    plt.savefig(f'/opt/{column}_distribution.png')

# Release cached data
data_app_train.unpersist()

# Stop SparkSession
spark.stop()
