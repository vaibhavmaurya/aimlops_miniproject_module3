import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import argparse
import os
import sys

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the parent directory
parent_dir = os.path.dirname(current_file_path)

# Get the parent's parent directory
grand_parent_dir = os.path.dirname(parent_dir)

# Add the grand parent directory to the sys.path
sys.path.insert(0, grand_parent_dir)


from bikeshare_model.datasets import DataSet
from bikeshare_model.pipeline import get_pipeline
from bikeshare_model.config import get_config

# print(f'''

#     MODEL_CONFIGURATION: {MODEL_CONFIGURATION}

# ''')



# Define a dictionary to map the model names in the YAML to the actual scikit-learn classes
MODEL_NAME_MAP = {
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "SGDRegressor": SGDRegressor
}



def load_dataset(dataset_config: dict) -> DataSet:
    dataset = DataSet(dataset_config)
    return dataset



def train_and_save_models(config: dict):
    """
    Function to train models based on configurations in YAML data,
    compute metrics, save models and metrics to disk.
    :param yaml_data: Dictionary object from parsed YAML data
    :return: metrics as a dictionary object
    """
    DATASET_CONFIGURATION = config.get("dataset_configuration", None)
    MODEL_CONFIGURATION = config.get("models", None)
    model_path = config.get("model_save_path", None)
    pipeline_save_path = config.get("pipeline_save_path", None)

    X_train, X_test, y_train, y_test = load_dataset(DATASET_CONFIGURATION).get_test_train_split()

    preprocess_pipeline = get_pipeline(DATASET_CONFIGURATION)

    X_train_transformed = preprocess_pipeline.fit_transform(X_train)

    # Save the pipeline to a file
    joblib.dump(preprocess_pipeline, os.path.join(pipeline_save_path, "preprocess_pipeline.joblib"))


    metrics = {}

    # For each model in the YAML...
    for model_info in MODEL_CONFIGURATION:
        model_name = model_info["model_name"]
        model_parameters = model_info["model_parameters"]

        # Find the corresponding scikit-learn model class
        ModelClass = MODEL_NAME_MAP[model_name]

        # Instantiate the model with the provided parameters
        model = ModelClass(**model_parameters)

        # Train the model
        model.fit(X_train_transformed, y_train.values.ravel())

        # Save the model to a file
        joblib.dump(model, os.path.join(model_path, f"{model_name}.joblib"))

        # Make predictions on the test data
        X_test_transformed = preprocess_pipeline.transform(X_test)
        y_pred = model.predict(X_test_transformed)

        # Compute metrics
        r2 = r2_score(y_test.values.ravel(), y_pred)
        mse = mean_squared_error(y_test.values.ravel(), y_pred)

        # Save the metrics
        metrics[model_name] = {"r2_score": r2, "mean_squared_error": mse}

    return metrics



def save_metrics(metrics: dict, metrics_path: str):
    """
    Function to save metrics to a JSON file.
    :param metrics: metrics as a dictionary object
    """
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)



def parse_arguments():
    """
    Function to parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a model based on provided config file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    return parser.parse_args()


def main(args):

    print("-----------Training Starts-----------\n")
    
    CONFIG = get_config(args.config)

    # global DATASET_CONFIGURATION
    # DATASET_CONFIGURATION = CONFIG.get("dataset_configuration", None)
    # global MODEL_CONFIGURATION
    # MODEL_CONFIGURATION = CONFIG.get("models", None)

    # global model_path
    # model_path = CONFIG.get("model_save_path", None)
    # global metrics_path
    metrics_path = CONFIG.get("model_metrics_save_path", None)
    
    metrics = train_and_save_models(CONFIG)
    save_metrics(metrics, metrics_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)


# def load_models_and_predict(yaml_data):
#     """
#     Function to load saved models, make predictions on test data.
#     :param yaml_data: Dictionary object from parsed YAML data
#     """
#     _, X_test, _, _ = load_data()

#     # Load the models and make predictions
#     for model_info in yaml_data["models"]:
#         model_name = model_info["model_name"]

#         # Load the model from a file
#         model = joblib.load(f"{model_name}.joblib")

#         # Make predictions on the test data
#         y_pred = model.predict(X_test)

#         print(f"Predictions for {model_name}: {y_pred[:5]}...")  # print the first 5 predictions

