Heart Disease
==============================

#### Project Installation

```
python -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
pip install -e.
```

#### Usage
Training

```
python src/train_pipeline.py configs/config_train_tree_k_fold_cv.yaml
```
```
python src/train_pipeline.py configs/config_train_logreg_train_test_split.yaml
```

Make prediction

```
python src/test_pipeline.py configs/config_test.yaml 
```

#### Test

```
pytest tests/
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile                    <- Makefile with commands like `make data` or `make train`
    ├── README.md                   <- The top-level README for developers using this project.
    ├── configs                     <- Configuration files
    ├── data
    │   ├── external                <- Data from third party sources.
    │   ├── interim                 <- Intermediate data that has been transformed.
    │   ├── processed               <- The final, canonical data sets for modeling.
    │   └── raw                     <- The original, immutable data dump.
    │
    ├── docs                        <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                      <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                   <- Jupyter notebooks. 
    │
    ├── predictions                 <- Files with predictions. 
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                               generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                    <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                         <- Source code for use in this project.
    │   ├── __init__.py             <- Makes src a Python module
    │   │
    │   ├── custom_transformers     <- Scripts to create custom transformers
    │   │   └── squared_features.py <- Script to create SquaredFeatures transformer
    │   │
    │   ├── data                    <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── enities                 <- Different objects and methods to store objects
    │   │   ├── _test_pipeline_params.py
    │   │   ├── feature_params.py
    │   │   ├── model_params.py
    │   │   ├── pipeline_params.py
    │   │   ├── train_pipeline_params.py
    │   │   └── validation_pipeline_params.py
    │   │
    │   ├── features                <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models                  <- Scripts to train models and then use trained models to make
    │   │   │                          predictions
    │   │   └── model_fit_predict.py
    │   │
    │   ├── test_pipeline.py        <- Script to test model 
    │   │
    │   └── train_pipeline.py       <- Script to train model   
    │
    ├── tests
    │   ├── configs                 <- Script to test configs parser
    │   │
    │   ├── configs_data            <- .yaml congig files for tests
    │   │
    │   ├── conftest.py             <- Configuration for tests
    │   │
    │   ├── data                    <- Scripts to test reading and splitting of the data
    │   │
    │   ├── datasets                <- Datasets for tests and skripts to generate datasets
    │   │
    │   ├── features                <- Scripts to test making and extracting features
    │   │
    │   ├── models                  <- Scripts to test training of models and predictions 
    │   │   
    │   └── test_end_to_end_training.py  <- Script to test the whole pipeline
    │
    ├── tox.ini                     <- tox file with settings for running tox; see tox.readthedocs.io
    │
    └── transformers                <- Fitted and dumped transformers


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

