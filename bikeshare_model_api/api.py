from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
from typing import Dict

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from parent_folder1 import module1

from bikeshare_model import get_metrics, final_prediction, get_config

app = FastAPI()

CONFIG = get_config("config/config.yml")

class PredictData(BaseModel):
    dteday: str
    season: str
    hr: str
    holiday: str
    weekday: str
    workingday: str
    weathersit: str
    temp: float
    atemp: float
    hum: int
    windspeed: float
    casual: int
    registered: int
    cnt: int



@app.post("/predict")
async def predict_api(data: PredictData):
    # data = predict_data.data
    predict_data = [data.dict()]
    print(predict_data)

    if not data:
        return JSONResponse(content={'message': 'No input data provided'}, status_code=400)

    predictions = final_prediction(predict_data, CONFIG)

    return JSONResponse(content=predictions)


@app.get("/metrics")
async def get_model_performance():
    try:
        return JSONResponse(content=get_metrics(CONFIG))
    except FileNotFoundError:
        return JSONResponse(content={'message': 'metrics.json not found'}, status_code=404)
