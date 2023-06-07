__all__ = ["final_prediction"]

import pandas as pd
from trained_models import load_models_and_predict

data = {'dteday': {'0': '2012-11-05', '1': '2011-07-13'},
 'season': {'0': 'winter', '1': 'fall'},
 'hr': {'0': '6am', '1': '4am'},
 'holiday': {'0': 'No', '1': 'No'},
 'weekday': {'0': 'Mon', '1': 'Wed'},
 'workingday': {'0': 'Yes', '1': 'Yes'},
 'weathersit': {'0': 'Mist', '1': 'Clear'},
 'temp': {'0': 6.1, '1': 26.78},
 'atemp': {'0': 3.0014, '1': 28.9988},
 'hum': {'0': 49.0, '1': 58.0},
 'windspeed': {'0': 19.0012, '1': 16.9979},
 'casual': {'0': 4, '1': 0},
 'registered': {'0': 135, '1': 5},
 'cnt': {'0': 139, '1': 5}}


def final_prediction(input_data):
    """
    Function to load saved models, make predictions on test data.
    :param yaml_data: Dictionary object from parsed YAML data
    """
    input_data = pd.DataFrame(input_data)
    output = load_models_and_predict(input_data)
    return output


def main():
    """
    Main function
    """
    input_data = pd.DataFrame(data)
    output = load_models_and_predict(input_data)
    print(output)


if __name__ == "__main__":
    main()


