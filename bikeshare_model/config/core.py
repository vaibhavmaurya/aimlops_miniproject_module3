import yaml
import os

script_directory = os.path.dirname(os.path.abspath(__file__))

with open(f"{script_directory}/config.yml", "r") as f:
    CONFIG = yaml.safe_load(f)


DATASET_CONFIGURATION = CONFIG.get("dataset_configuration", None)
MODEL_CONFIGURATION = CONFIG.get("models", None)

__all__ = ["CONFIG", "DATASET_CONFIGURATION", "MODEL_CONFIGURATION"]