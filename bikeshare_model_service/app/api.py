import json
from typing import Any
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from bikeshare_model.predict import final_prediction
from bikeshare_model.trained_models import get_metrics
import schemas
from service_config import settings


api_router = APIRouter()

@api_router.post('/predict', response_model=schemas.PredictionResults, status_code=200 )
def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    This function takes in a POST request with JSON data containing the 
    input for the models. The function applies the models to the data 
    and returns a JSON response with the input data and the predictions.

    Returns:
    - 400 status if no data is provided.
    - A JSON response containing the input data and the model predictions.
    """
    results = final_prediction(jsonable_encoder(input_data.inputs))

    if results.get("errors") is not None:
        raise HTTPException(status_code=400, detail=json.loads(results.get("errors")))

    return results


@api_router.get('/metrics', response_model=schemas.Metrics, status_code=200)
async def metrics():
    """
    This function takes in a GET request and returns a JSON response 
    containing the contents of the 'metrics.json' file.

    Returns:
    - 404 status if the file is not found.
    - A JSON response containing the contents of the 'metrics.json' file.
    """
    try:
        # Return the JSON data as a response
        return get_metrics()
    
    except FileNotFoundError:
        # Return a 404 status if the file is not found
        return {'message': 'metrics.json not found'}, 404
