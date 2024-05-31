from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import logging

spark = SparkSession.builder \
    .appName("Credit Risk Modeling") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

data_path = "hdfs://namenode:9000/home_credit/data/"
parquet_path = "hdfs://namenode:9000/home_credit/data_parquet/"


def convert_csv_to_parquet(file_name):
    df = spark.read.csv(data_path + file_name, header=True, inferSchema=True)
    df.write.parquet(parquet_path + file_name.replace('.csv',
                     '.parquet'), mode='overwrite')


convert_csv_to_parquet("application_train.csv")
convert_csv_to_parquet("application_test.csv")

logging.basicConfig(level=logging.INFO)


def read_and_show_parquet(file_name):
    df = spark.read.parquet(parquet_path + file_name)
    logging.info(df.show())
    return df


data_app_train = read_and_show_parquet("application_train.parquet")
data_app_test = read_and_show_parquet("application_test.parquet")

data = data_app_train.sample(fraction=0.1, seed=1234).drop(
    'SK_ID_CURR').withColumnRenamed('TARGET', 'label')
data.persist()  # Persist data to memory

categorical_columns = [t[0] for t in data.dtypes if t[1] == 'string']
numerical_columns = [t[0]
                     for t in data.dtypes if t[1] != 'string' and t[0] != 'label']
data = data.fillna('unknown', subset=categorical_columns).fillna(
    0, subset=numerical_columns)

(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=1234)

indexers = [StringIndexer(inputCol=col, outputCol=col + "_indexed",
                          handleInvalid='keep') for col in categorical_columns]
encoders = [OneHotEncoder(inputCol=idx.getOutputCol(
), outputCol=idx.getOutputCol() + "_encoded") for idx in indexers]

for model, name in [(LogisticRegression(featuresCol="features", labelCol="label"), "Logistic Regression"), (RandomForestClassifier(featuresCol="features", labelCol="label"), "Random Forest")]:
    assembler = VectorAssembler(inputCols=[enc.getOutputCol(
    ) for enc in encoders] + numerical_columns, outputCol="features")
    pipeline = Pipeline(stages=indexers + encoders + [assembler, model])
    fitted_model = pipeline.fit(train_data)
    predictions = fitted_model.transform(test_data)

    # Đánh giá độ chính xác
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_accuracy.evaluate(predictions)
    predictions.unpersist()  # Giải phóng bộ nhớ

    # Đánh giá F1 Score
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator_f1.evaluate(predictions)
    predictions.unpersist()  # Giải phóng bộ nhớ

    # Ghi thông tin vào file log bằng logger mới
    logging.info(f"Model: {name}")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"F1 Score: {f1_score}")

train_data.unpersist()  # Giải phóng bộ nhớ
test_data.unpersist()  # Giải phóng bộ nhớ
