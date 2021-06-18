import os
import pathlib


import pandas as pd
import numpy as np
import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

from utils import load, _wait_for_file



def _predict(
        data_fp: str,
        model_fp: str,
        output_dir: str,
):
    output_fp = os.path.join(output_dir, 'predictions.csv')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_fp)
    model = load(model_fp)
    preds = model.predict(data)
    df_preds = pd.DataFrame(preds, columns=['prediction'])
    df_preds.to_csv(output_fp, index=False)
    print(f'Predictions stored in {output_fp}')


_predict("data/processed/2021-06-09/test_data.csv", "models/2021-06-09/model.pkl", "abiba")


with DAG(
    dag_id='predict',
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@daily',
    max_active_runs=1,
) as dag:
    data_sensor = PythonSensor(
        task_id='data_sensor',
        python_callable=_wait_for_file,
        op_kwargs={'path': '/opt/airflow/data/processed/{{ ds }}/test_data.csv'},
        timeout=60,
        poke_interval=10,
        retries=100,
        mode='poke',
    )

    model_sensor = PythonSensor(
        task_id='model_sensor',
        python_callable=_wait_for_file,
        op_kwargs={'path': '{{ var.value.model_fp }}'},
        timeout=60,
        poke_interval=10,
        retries=100,
        mode='poke',
    )

    predict = PythonOperator(
        task_id='predict',
        python_callable=_predict,
        op_kwargs={
            'data_fp': '/opt/airflow/data/processed/{{ ds }}/test_data.csv',
            'model_fp': '{{ var.value.model_fp }}',
            'output_dir': '/opt/airflow/data/predictions/{{ ds }}/',
        }
    )

    [data_sensor, model_sensor] >> predict