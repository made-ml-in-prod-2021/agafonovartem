import sys
import pytest
from airflow.models import DagBag


sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder='dags/', include_examples=False)


def test_dag_generate_data(dag_bag):
    assert 'data_generator' in dag_bag.dags
    dag = dag_bag.dags['data_generator']
    assert len(dag.tasks) == 2


def test_dag_generate_data_structure(dag_bag):
    dag = dag_bag.dags['data_generator']

    structure = {
        'data_generator': ['bash_command'],
        'bash_command': []
    }
    for name, task in dag.task_dict.items():
        downstreams = task.downstream_task_ids
        assert downstreams == set(structure[name])


def test_dag_train(dag_bag):
    assert 'train_validate' in dag_bag.dags
    dag = dag_bag.dags['train_validate']
    assert len(dag.tasks) == 5
    assert 'data_sensor' in dag.task_dict


def test_dag_train_stucture(dag_bag):
    dag = dag_bag.dags['train_validate']
    structure = {
        'data_sensor': ['train_test_split'],
        'target_sensor': ['train_test_split'],
        'train_test_split': ['train_model'],
        'train_model': ['validate_model'],
        'validate_model': []
    }
    for name, task in dag.task_dict.items():
        downstreams = task.downstream_task_ids
        assert downstreams == set(structure[name])


def test_dag_predict(dag_bag):
    assert 'predict' in dag_bag.dags
    dag = dag_bag.dags['predict']
    assert len(dag.tasks) == 3
    assert 'data_sensor' in dag.task_dict


def test_dag_predict_stucture(dag_bag):
    dag = dag_bag.dags['predict']
    structure = {
        'data_sensor': ['predict'],
        'model_sensor': ['predict'],
        'predict': []
    }
    for name, task in dag.task_dict.items():
        downstreams = task.downstream_task_ids
        assert downstreams == set(structure[name])