from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql import SparkSession
import xgboost as xgb
import logging

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Credit Risk Modeling") \
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
data_app_test = read_and_show_parquet("application_test.parquet")

# Sample and preprocess data
data = data_app_train.sample(fraction=0.1, seed=1234).drop(
    'SK_ID_CURR').withColumnRenamed('TARGET', 'label')
data.persist()  # Cache data in memory

categorical_columns = [t[0] for t in data.dtypes if t[1] == 'string']
numerical_columns = [t[0]
                     for t in data.dtypes if t[1] != 'string' and t[0] != 'label']
data = data.fillna('unknown', subset=categorical_columns).fillna(
    0, subset=numerical_columns)

# Split data into train and test sets
(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=1234)

# Prepare transformation stages
indexers = [StringIndexer(inputCol=col, outputCol=col + "_indexed",
                          handleInvalid='keep') for col in categorical_columns]
encoders = [OneHotEncoder(inputCol=idx.getOutputCol(),
                          outputCol=idx.getOutputCol() + "_encoded") for idx in indexers]

# Set up Pipeline with VectorAssembler
assembler = VectorAssembler(inputCols=[enc.getOutputCol(
) for enc in encoders] + numerical_columns, outputCol="features")
pipeline_stages = indexers + encoders + [assembler]

# Apply pipeline to training data
pipeline = Pipeline(stages=pipeline_stages)
pipeline_model = pipeline.fit(train_data)
train_data_transformed = pipeline_model.transform(train_data)
test_data_transformed = pipeline_model.transform(test_data)

# Extract features and labels
X_train = train_data_transformed.select("features").rdd.map(
    lambda row: row[0].toArray()).collect()
y_train = train_data_transformed.select(
    "label").rdd.map(lambda row: row[0]).collect()
X_test = test_data_transformed.select("features").rdd.map(
    lambda row: row[0].toArray()).collect()
y_test = test_data_transformed.select(
    "label").rdd.map(lambda row: row[0]).collect()

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Train XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 6,
    'eta': 0.01,
    'objective': 'binary:logistic',
    'colsample_bytree': 0.5,
    'seed': 12345
}
num_round = 100
bst = xgb.train(param, dtrain, num_round)

# Predict and evaluate
preds = bst.predict(dtest)
predictions = [round(value) for value in preds]


accuracy_xgb = accuracy_score(y_test, predictions)
f1_score_xgb = f1_score(y_test, predictions)

logging.info("XGBoost Model:")
logging.info(f"Accuracy: {accuracy_xgb}")
logging.info(f"F1 Score: {f1_score_xgb}")

# Release cached data
data.unpersist()

# Stop SparkSession
spark.stop()
