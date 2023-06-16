from pydantic import BaseModel

class Model(BaseModel):
    r2_score: float
    mean_squared_error: float

class Metrics(BaseModel):
    LinearRegression: Model
    RandomForestRegressor: Model
    SGDRegressor: Model