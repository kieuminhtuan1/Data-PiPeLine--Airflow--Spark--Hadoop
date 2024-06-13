from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
encoders = [OneHotEncoder(inputCol=idx.getOutputCol(
), outputCol=idx.getOutputCol() + "_encoded") for idx in indexers]

# Initialize the Logistic Regression model
logistic_regression = LogisticRegression(
    featuresCol="features", labelCol="label")

# Set up Pipeline with VectorAssembler and Logistic Regression model
assembler = VectorAssembler(inputCols=[enc.getOutputCol(
) for enc in encoders] + numerical_columns, outputCol="features")
pipeline = Pipeline(stages=indexers + encoders +
                    [assembler, logistic_regression])

# Train and predict with Logistic Regression model
fitted_model = pipeline.fit(train_data)
predictions = fitted_model.transform(test_data)

# Evaluate model accuracy
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(predictions)

# Evaluate model F1 Score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)

# Log model performance
logging.info(f"Model: Logistic Regression")
logging.info(f"Accuracy: {accuracy}")
logging.info(f"F1 Score: {f1_score}")

# Release cached data
data.unpersist()
train_data.unpersist()
test_data.unpersist()

# Stop SparkSession
spark.stop()
