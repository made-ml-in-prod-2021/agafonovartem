## REST API & ML
#### Downloading  docker image
```
docker pull fiztehno/online_inference:v2 
docker run -p 8000:8000 fiztehno/online_inference:v2
```
#### Building from scratch
```
docker build -t fiztehno/online_inference:v2 .
docker run -p 8000:8000 fiztehno/online_inference:v2
```
#### Making requests
```
python make_request.py 
```
#### Tests
```
pytest tests
```
### Project Organization
    ├── app.py                      <- Main file with the app 
    │
    ├── make_requests.py            <- Script to make requests 
    │ 
    ├── Dockerfile                  <- Docker file, that contains all the commands to assemble an image
    │
    ├── README.md                   <- The top-level README for developers using this project.
    │
    ├── data
    │   └── raw                     <- The original, immutable data dump.
    │
    ├── models                      <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                   <- Jupyter notebooks. 
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
    │   └── utils.py                <- Script with common functions and classes. 
    │
    ├── tests
    │   │
    │   ├── data                    <- Datasets for tests
    │   │   
    │   └── test_app.py             <- Script to test the whole app
    │
    └── transformers                <- Fitted and dumped transformers

#### Docker image size optimization

Python-slim is used instead of standard python. It saves 711 Mb.
Also tried to use python-alpine, but had not waited for the image creation to complete.