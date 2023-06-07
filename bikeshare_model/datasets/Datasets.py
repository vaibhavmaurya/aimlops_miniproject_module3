__all__ = ['DataSet']

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import os


script_directory = os.path.dirname(os.path.abspath(__file__))

class DataSet(pd.DataFrame):

  def __init__(self, dataset_config):
    super().__init__(pd.read_csv(f'''{script_directory}/{dataset_config["csv_file_path"]}'''))
    self.dataset_config = dataset_config
    # Target_feature


  def extract_year_month(self):
    """
    Assuming the date format is 'yyyy-mm-dd'

    Args:
    date_series: A Pandas Series of dates in the format 'yyyy-mm-dd'.
    """
    date_series_todate = pd.to_datetime(self['dteday'], format='%Y-%m-%d')
    self['yr'] = date_series_todate.dt.year
    self['mnth'] = date_series_todate.dt.month_name()
    # self.drop(columns='dteday', inplace=True) 
    return self

  
  def get_test_train_split(self, test_size=0.2):
    """
    Return test and train split of the data

    Args:
    test_size: test size ratio
    """    
    input_features = list(self.columns)
    input_features.remove(self.dataset_config["Target_feature"])
    X_train, X_test, y_train, y_test = train_test_split(self[input_features],
                                                        self[self.dataset_config["Target_feature"]],
                                                        test_size=test_size,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test