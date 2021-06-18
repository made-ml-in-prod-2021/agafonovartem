import os
import pathlib


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

from utils import load, dump, calculate_metrics, _wait_for_file


RANDOM_STATE = 52
TEST_SIZE = 0.2


def _train_test_split(data_dir: str,
                      execution_date: str,
                      test_size: float,
                      random_state: int):
    output_dir = os.path.join(data_dir, 'processed', execution_date)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    data_fp = os.path.join(data_dir, 'raw', execution_date, 'data.csv')
    target_fp = os.path.join(data_dir, 'raw', execution_date, 'target.csv')

    data = pd.read_csv(data_fp)
    target = pd.read_csv(target_fp)

    x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=test_size,
                                                        random_state=random_state)

    x_train.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    x_test.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'train_target.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'test_target.csv'), index=False)

    print(f"Split data stored in {output_dir}")


def _train_model(processed_data_dir: str,
                 output_model_dir: str,
                 model_name: str
                ):

    train_data_fp = os.path.join(processed_data_dir, 'train_data.csv')
    train_target_fp = os.path.join(processed_data_dir, 'train_target.csv')

    train_data = pd.read_csv(train_data_fp)
    train_target = pd.read_csv(train_target_fp)

    model = LinearRegression().fit(train_data, train_target)

    pathlib.Path(output_model_dir).mkdir(parents=True, exist_ok=True)

    output_model_fp = os.path.join(output_model_dir, model_name)
    dump(model, output_model_fp)


    print(f"LinearRegression stored in {output_model_fp}")


def _validate_model(processed_data_dir: str,
                    model_fp: str,
                    output_metrics_dir: str):

    test_data_fp = os.path.join(processed_data_dir, 'test_data.csv')
    test_target_fp = os.path.join(processed_data_dir, 'test_target.csv')

    test_data = pd.read_csv(test_data_fp)
    test_target = pd.read_csv(test_target_fp)

    model = load(model_fp)
    preds = model.predict(test_data)
    metrics = calculate_metrics(test_target, preds)

    pathlib.Path(output_metrics_dir).mkdir(parents=True, exist_ok=True)

    output_metrics_fp = os.path.join(output_metrics_dir, 'metrics.pkl')
    print(f"Metrics on validation{metrics}")
    print(f"Metrics are stored in {output_metrics_fp}")
    dump(metrics, output_metrics_fp)


with DAG(
    dag_id='train_validate',
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@weekly',
    max_active_runs=1,
) as dag:
    data_sensor = PythonSensor(
        task_id='data_sensor',
        python_callable=_wait_for_file,
        op_kwargs={'path': '/opt/airflow/data/raw/{{ ds }}/data.csv'},
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode='poke',
    )

    target_sensor = PythonSensor(
        task_id='target_sensor',
        python_callable=_wait_for_file,
        op_kwargs={'path': '/opt/airflow/data/raw/{{ ds }}/target.csv'},
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode='poke',
    )

    train_test_split_ = PythonOperator(
        task_id='train_test_split',
        python_callable=_train_test_split,

        op_kwargs={
            'execution_date': '{{ ds }}',
            'data_dir': '/opt/airflow/data/',
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
        }
    )


    train_model = PythonOperator(
        task_id='train_model',
        python_callable=_train_model,
        op_kwargs={
            'processed_data_dir': '/opt/airflow/data/processed/{{ ds }}',
            'output_model_dir': '/opt/airflow/models/{{ ds }}',
            'model_name': 'model.pkl'
        }
    )


    validate_model = PythonOperator(
        task_id='validate_model',
        python_callable=_validate_model,
        op_kwargs={
            'processed_data_dir': '/opt/airflow/data/processed/{{ ds }}',
            'model_fp': '/opt/airflow/models/{{ ds }}/model.pkl',
            'output_metrics_dir': '/opt/airflow/metrics/{{ ds }}',
        }
    )

    [data_sensor, target_sensor] >> train_test_split_ >> train_model  >> validate_model












