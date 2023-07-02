import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import pytest
# from sklearn.model_selection import train_test_split
import pandas as pd
# from bikeshare_model.processing.data_manager import load_dataset

# /mnt/c/Users/91961/Documents/Learn/AIML/MLOps/MiniProject/Module3/config/config.yml

@pytest.fixture(scope="session")
def sample_input_data():
    sample_data = [{"dteday":"2011-07-13","season":"fall","hr":"4am","holiday":"No","weekday":"Wed","workingday":"Yes","weathersit":"Clear","temp":26.78,"atemp":28.9988,"hum":58.0,"windspeed":16.9979,"casual":0,"registered":5,"cnt":5},{"dteday":"2012-02-09","season":"spring","hr":"11am","holiday":"No","weekday":"Thu","workingday":"Yes","weathersit":"Clear","temp":3.28,"atemp":-0.9982,"hum":52.0,"windspeed":15.0013,"casual":4,"registered":95,"cnt":99}]
    X_test = pd.DataFrame(sample_data)
    return X_test