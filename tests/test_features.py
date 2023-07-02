
"""
Note: These tests will fail if you have not first trained the model.
"""

# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
# from bikeshare_model.config import DATASET_CONFIGURATION
from bikeshare_model.processing import WeekdayImputer, WeathersitImputer, Mapper, OutlierHandler, WeekdayOneHotEncoder


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



# def test_weekday_variable_imputer(sample_input_data):
#     # Given
#     imputer = WeekdayImputer(variable = config.model_config.weekday_var, date_var = config.model_config.date_var)
#     assert np.isnan(sample_input_data[0].loc[7046, 'weekday'])

#     # When
#     subject = imputer.fit(sample_input_data[0]).transform(sample_input_data[0])

#     # Then
#     assert subject.loc[7046, 'weekday'] == 'Wed'


# def test_weathersit_variable_imputer(sample_input_data):
#     # Given
#     imputer = WeathersitImputer(variable = config.model_config.weathersit_var)
#     assert np.isnan(sample_input_data[0].loc[7046, 'weathersit'])

#     # When
#     subject = imputer.fit(sample_input_data[0]).transform(sample_input_data[0])

#     # Then
#     assert subject.loc[7046, 'weathersit'] == 'Clear'


# def test_season_variable_mapper(sample_input_data):
#     # Given
#     mapper = Mapper(variable = config.model_config.season_var, 
#                     mappings = config.model_config.season_mappings)
#     assert sample_input_data[0].loc[8688, 'season'] == 'summer'

#     # When
#     subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

#     # Then
#     assert subject.loc[8688, 'season'] == 2


# def test_windspeed_variable_outlierhandler(sample_input_data):
#     # Given
#     encoder = OutlierHandler(variable = config.model_config.windspeed_var)
#     q1, q3 = np.percentile(sample_input_data[0]['windspeed'], q=[25, 75])
#     iqr = q3 - q1
#     assert sample_input_data[0].loc[5813, 'windspeed'] > q3 + (1.5 * iqr)

#     # When
#     subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

#     # Then
#     assert subject.loc[5813, 'windspeed'] <= q3 + (1.5 * iqr)


# def test_weekday_variable_encoder(sample_input_data):
#     # Given
#     encoder = WeekdayOneHotEncoder(variable = config.model_config.weekday_var)
#     assert sample_input_data[0].loc[8688, 'weekday'] == 'Sun'

#     # When
#     subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

#     # Then
#     assert subject.loc[8688, 'weekday_Sun'] == 1.0

