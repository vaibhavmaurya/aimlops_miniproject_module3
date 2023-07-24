__all__ = ['WeekdayImputer', 'WeathersitImputer', 'YearMonthImputer']

from typing import Dict
import ray
from pandas import DataFrame
import pandas as pd
from ray.data.preprocessor import Preprocessor
from ray.data import Dataset
from ray.data.aggregate import Max



    
def yearMonthColumnAdd(batch: DataFrame) -> DataFrame:
    date_series_todate = pd.to_datetime(batch['dteday'], format='%Y-%m-%d')
    batch['yr'] = date_series_todate.dt.year
    batch['mnth'] = date_series_todate.dt.month_name()
    return batch

def weekDayImputerfn(batch: DataFrame) -> DataFrame:
    dteday = batch['dteday']
    weekday = pd.to_datetime(dteday, format='%Y-%m-%d').dt.day_name().apply(lambda x : x[:3])
    batch['weekday'] = batch['weekday'].fillna(weekday)
    return batch

def removeUnusedColumns(batch: DataFrame) -> DataFrame:
    unused_colms = ['dteday', 'casual', 'registered', 'weekday']
    batch.drop(labels = unused_colms, axis = 1, inplace = True)
    return batch