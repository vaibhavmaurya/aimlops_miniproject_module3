__all__ = ['Mapper']

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class Mapper(BaseEstimator, TransformerMixin):
    """
    This class is used for mapping ordinal categorical variables in a pandas DataFrame.

    Each mapping dictionary represents a categorical variable and its corresponding ordinal mapping. 
    The mapping is defined in the __init__ method of the class.

    Attributes
    ----------
    mapping : dict
        A dictionary that contains mapping for each ordinal categorical variable.

    Methods
    -------
    fit(X: pd.DataFrame):
        This function is a placeholder to satisfy sklearn's TransformerMixin interface.

    transform(X: pd.DataFrame) -> pd.DataFrame:
        This function transforms the pandas DataFrame by replacing the ordinal categorical variables 
        according to the predefined mappings.
    """

    def __init__(self):
        yr_mapping = {2011: 0, 2012: 1}
        mnth_mapping = {'January': 0, 'February': 1, 'December': 2, 
                             'March': 3, 'November': 4, 'April': 5, 
                             'October': 6, 'May': 7, 'September': 8, 
                             'June': 9, 'July': 10, 'August': 11}

        season_mapping = {'spring': 0, 'winter': 1, 'summer': 2,
                               'fall': 3}

        weather_mapping = {'Heavy Rain': 0, 'Light Rain': 1, 
                                'Mist': 2, 'Clear': 3}

        workingday_mapping = {'No': 0, 'Yes': 1}

        hour_mapping = {'4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, 
                             '12am': 5, '6am': 6, '11pm': 7, '10pm': 8, 
                '10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, 
                '8pm': 14, '2pm': 15, '1pm': 16, 
                '12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, 
                '6pm': 22, '5pm': 23}
        holiday_mapping = {'Yes': 0, 'No': 1}

        self.mapping = {
            'yr'        :yr_mapping,
            'mnth'      :mnth_mapping,
            'season'    :season_mapping,
            'weathersit':weather_mapping,
            'holiday'   :holiday_mapping,
            'workingday':workingday_mapping,
            'hr'        :hour_mapping
        }


    def fit(self, X: pd.DataFrame, y=None):
        """
        This function is a placeholder to satisfy sklearn's TransformerMixin interface.
        In this specific implementation, it does not perform any action.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame. 
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        This function transforms the pandas DataFrame by replacing the ordinal categorical variables 
        according to the predefined mappings.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame. 

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame where the categorical variables have been replaced by their ordinal 
            counterparts according to the predefined mappings.
        """
        for column, map_dict in self.mapping.items():
          try:
            X[column] = X[column].apply(lambda x: map_dict[x])
          except Exception as e:
            print(f'''
              Mapper: {column}  is missing
              exception is : {e}
            ''')
            raise Exception(e)
        return X
        