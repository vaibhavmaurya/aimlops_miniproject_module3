
"""
Note: These tests will fail if you have not first trained the model.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# # Get the absolute path of the current file
current_file_path = str(Path(__file__).parent.parent.parent)

sys.path.append(current_file_path)

from bikeshare_model.config import get_config
from bikeshare_model.processing import WeekdayImputer, WeathersitImputer, Mapper, OutlierHandler, WeekdayOneHotEncoder

## load the config

config = get_config( "config.yml")

def test_weekday_variable_imputer(sample_input_data):
    # Given
    X_test = sample_input_data

    # print(X_test)

    imputer = WeekdayImputer()
    assert not pd.isna(X_test.loc[0, 'weekday'])
    print(f'''
          X_test.loc[7091, 'weekday'] = {X_test.loc[0, 'weekday']}''')
    # When
    subject = imputer.fit(X_test).transform(X_test)

    # Then
    assert subject.loc[0, 'weekday'] == 'Wed'
    print(f'''
          subject.loc[7091, 'weekday'] = {subject.loc[0, 'weekday']}''')


def test_weekday_variable_encoder(sample_input_data):
      DATASET_CONFIGURATION = config.get("dataset_configuration", None)
      
      assert sample_input_data.iloc[0]['weekday'] == 'Wed'      

      encoder = WeekdayOneHotEncoder( DATASET_CONFIGURATION["categorical_variables"][3])
      
      # When
      subject = encoder.fit(sample_input_data).transform(sample_input_data)
      
      # Then    
      assert subject.iloc[1][ 'weekday_Thu'] == 1.0

