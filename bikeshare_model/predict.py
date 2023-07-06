__all__ = ["final_prediction"]

import os
import sys
import argparse
import pandas as pd

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the parent directory
parent_dir = os.path.dirname(current_file_path)

# Get the parent's parent directory
grand_parent_dir = os.path.dirname(parent_dir)

# Add the grand parent directory to the sys.path
sys.path.insert(0, grand_parent_dir)

from bikeshare_model.trained_models import load_models_and_predict
from bikeshare_model.config import get_config

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


def final_prediction(input_data, config):
    """
    Function to load saved models, make predictions on test data.
    :param yaml_data: Dictionary object from parsed YAML data
    """
    input_data = pd.DataFrame(input_data)
    output = load_models_and_predict(input_data, config)
    return output

def parse_arguments():
    """
    Function to parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a model based on provided config file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    return parser.parse_args()

def main(args):
    """
    Main function
    """
    
    CONFIG = get_config(args.config)
    
    input_data = pd.DataFrame(data)
    output = load_models_and_predict(input_data, CONFIG)
    print(output)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)


