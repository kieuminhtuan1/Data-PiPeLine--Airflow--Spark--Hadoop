from airflow import DAG
from airflow.operators.bash_operator import BashOperator
import datetime as dt

upload_command = """
docker exec -i namenode bash -c 'hdfs dfs -mkdir -p /home_credit/data && hdfs dfs -put -f /opt/data/*.parquet hdfs://namenode:9000/home_credit/data/'
"""

default_args = {
    'owner': 'FINAL',
    'start_date': dt.datetime.now(),
    'retries': 3,
    'retry_delay': dt.timedelta(minutes=5),
}

with DAG('final', default_args=default_args, schedule_interval='@daily') as dag:
    upload_task = BashOperator(
        task_id='upload_to_hdfs_task', bash_command=upload_command)
    EDAa = BashOperator(
        task_id='EDA', bash_command="docker exec spark-master /spark/bin/spark-submit --master spark://spark-master:7077 /opt/scripts/pyspark/EDA.py")
    Logistics_model = BashOperator(
        task_id='Logistics_model', bash_command="docker exec spark-master /spark/bin/spark-submit --master spark://spark-master:7077 /opt/scripts/pyspark/Logistics_model.py")
    Random_forest_model = BashOperator(
        task_id='Random_forest_model', bash_command="docker exec spark-master /spark/bin/spark-submit --master spark://spark-master:7077 /opt/scripts/pyspark/Random_forest_model.py")
    XGBoost_model = BashOperator(
        task_id='XGBoost_model', bash_command="docker exec spark-master /spark/bin/spark-submit --master spark://spark-master:7077 /opt/scripts/pyspark/XGBoost_model.py")
    upload_task >> EDAa >> [Logistics_model,
                            Random_forest_model, XGBoost_model]
