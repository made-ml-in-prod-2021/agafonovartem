import os
import pathlib


import pandas as pd
import numpy as np
import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


DATASET_SIZE = 100
NUM_FEATURES = 5


def _data_generator(
        output_dir: str,
        execution_date: str,
        num_features: int,
        dataset_size: int,
):

    assert dataset_size > 0, (
        f"Wrong dataset size!"
    )
    assert num_features > 0, (
        f"Wrong num_features size!"
    )

    pathlib.Path(os.path.join(output_dir, execution_date)).mkdir(parents=True, exist_ok=True)
    output_data_fp = os.path.join(output_dir, execution_date, 'data.csv')
    output_target_fp = os.path.join(output_dir, execution_date, 'target.csv')

    size = (dataset_size, num_features)

    feature_columns = list(map(str, range(1, num_features)))
    columns = feature_columns.copy()
    columns.append("target")


    df = pd.DataFrame(np.random.randint(0, 100, size=size),
                      columns=columns)

    df["target"].to_csv(output_target_fp, index=False)
    df[feature_columns].to_csv(output_data_fp, index=False)

    print(f"Data filepath {output_data_fp}, target filepath {output_target_fp}")


with DAG(
    dag_id='data_generator',
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@daily',
    max_active_runs=1,
) as dag:
    generate_data = PythonOperator(
        task_id='data_generator',
        python_callable=_data_generator,
        op_kwargs={
            'output_dir': '/opt/airflow/data/raw/',
            'execution_date': '{{ ds }}',
            'num_features': NUM_FEATURES,
            'dataset_size': DATASET_SIZE,
        }
    )

    endpoint = BashOperator(
        task_id='bash_command',
        bash_command='echo "DAG was successfully finished and all data was saved in /data/raw/{{ ds }}/"',
    )

    generate_data >> endpoint
