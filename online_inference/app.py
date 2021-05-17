import logging
import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException


from src.utils import (SklearnClassificationModel, ColumnTransformer,
                       HeartDiseaseResponse, HeartDiseaseModel,
                       load_object, make_predict, validate_features_same_order,
                       validate_features_right_type)

logger = logging.getLogger(__name__)

app = FastAPI()


model: Optional[SklearnClassificationModel] = None
transformer: Optional[ColumnTransformer] = None


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model, transformer
    model_path = os.getenv("PATH_TO_MODEL")
    transformer_path = os.getenv("PATH_TO_TRANSFORMER")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    if transformer_path is None:
        err = f"PATH_TO_TRANSFORMER {transformer_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)
    transformer = load_object(transformer_path)


@app.get("/healz/model")
def health() -> bool:
    return not (model is None)


@app.get("/healz/transformer")
def health() -> bool:
    return not (transformer is None)


@app.get("/predict/", response_model=List[HeartDiseaseResponse])
def predict(request: HeartDiseaseModel) -> List[HeartDiseaseResponse]:
    if not validate_features_same_order(request.features):
        raise HTTPException(status_code=400, detail=f"Wrong feature order!")
    if not validate_features_right_type(request):
        raise HTTPException(status_code=400, detail=f"Wrong data values!")
    return make_predict(request.data, request.features, model, transformer)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
