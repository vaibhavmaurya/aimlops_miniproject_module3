__all__ = ['WeekdayImputer', 'WeathersitImputer', 'YearMonthImputer']

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class YearMonthImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self):
        self.dteday = None
        self.weekday = None

    def fit(self, X: pd.DataFrame, y=None):
        """ Fits the imputer by extracting the 'dteday' and 'weekday' columns from the DataFrame. """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Imputes the missing values in the 'weekday' column. """
        date_series_todate = pd.to_datetime(X['dteday'], format='%Y-%m-%d')
        X['yr'] = date_series_todate.dt.year
        X['mnth'] = date_series_todate.dt.month_name()
        return X


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self):
        self.dteday = None
        self.weekday = None

    def fit(self, X: pd.DataFrame, y=None):
        """ Fits the imputer by extracting the 'dteday' and 'weekday' columns from the DataFrame. """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Imputes the missing values in the 'weekday' column. """
        self.dteday = X['dteday']
        self.weekday = pd.to_datetime(self.dteday, format='%Y-%m-%d').dt.day_name().apply(lambda x : x[:3])
        X['weekday'] = X['weekday'].fillna(self.weekday)
        # X.drop(columns='dteday', inplace=True)
        return X 
    


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self):
        self.weathersit = None

    def fit(self, X: pd.DataFrame, y=None):
        """ Fits the imputer by extracting the 'weathersit' """
        self.weathersit = X['weathersit']
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Imputes the missing values in the 'weekday' column. """
        X['weathersit'] = X['weathersit'].fillna('Clear')
        return X  