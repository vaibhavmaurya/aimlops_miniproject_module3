from typing import Any, List, Optional

from pydantic import BaseModel

class DataInputSchema(BaseModel):
    dteday: str
    season: str
    hr: str
    holiday: str
    weekday: str
    workingday: str
    weathersit: str
    temp: float
    atemp: float
    hum: float
    windspeed: float
    casual: int
    registered: int
    cnt: int

class PredictionResults(BaseModel):
    LinearRegression: List[float]
    RandomForestRegressor: List[float]
    SGDRegressor: List[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

class Config:
    schema_extra = {
        "example": {
            "inputs": [
                {
                    "PassengerId": 79,
                    "Pclass": 2,
                    "Name": "Caldwell, Master. Alden Gates",
                    "Sex": "male",
                    "Age": 0.83,
                    "SibSp": 0,
                    "Parch": 2,
                    "Ticket": "248738",
                    "Cabin": 'A5',
                    "Embarked": "S",
                    "Fare": 29,
                }
            ]
        }
    }