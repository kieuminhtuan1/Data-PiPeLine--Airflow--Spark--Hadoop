from airflow import DAG
from airflow.operators.bash_operator import BashOperator
import datetime as dt

upload_command = """
docker exec -i namenode bash -c 'hdfs dfs -mkdir -p /home_credit/data && hdfs dfs -put -f /opt/data/*.csv hdfs://namenode:9000/home_credit/data/'
"""

default_args = {
    'owner': 'FINAL',
    'start_date': dt.datetime.now(),
    'retries': 3,
    'retry_delay': dt.timedelta(minutes=5),
}

with DAG('final', default_args=default_args, schedule_interval='@hourly') as dag:
    upload_task = BashOperator(
        task_id='upload_csv_to_hdfs_task', bash_command=upload_command)
    run_spark = BashOperator(
        task_id='run_spark', bash_command="docker exec spark-master /spark/bin/spark-submit --master spark://spark-master:7077 /opt/scripts/pyspark/sample_spark_2.py")
    upload_task >> run_spark
