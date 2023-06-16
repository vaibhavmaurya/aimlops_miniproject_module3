import yaml
import os


def get_config(path: str):
    print(f'''
           Path to config file: {path}
              ''')
    if not (os.path.exists(path) and os.path.isfile(path)):
        raise FileNotFoundError(f"File not found at {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


__all__ = ["get_config"]