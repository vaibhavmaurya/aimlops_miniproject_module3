from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from bikeshare_model import get_metrics, final_prediction, get_config

app = FastAPI()

CONFIG = get_config("/mnt/c/Users/91961/Documents/Learn/AIML/MLOps/MiniProject/Module3/config/config.yml")

class PredictData(BaseModel):
    data: list

@app.post("/predict")
async def predict_api(predict_data: PredictData):
    data = predict_data.data

    if not data:
        return JSONResponse(content={'message': 'No input data provided'}, status_code=400)

    predictions = final_prediction(data, CONFIG)

    response = {
        'input': data,
        'predictions': predictions
    }

    return JSONResponse(content=response)


@app.get("/metrics")
async def get_model_performance():
    try:
        return JSONResponse(content=get_metrics(CONFIG))
    except FileNotFoundError:
        return JSONResponse(content={'message': 'metrics.json not found'}, status_code=404)
