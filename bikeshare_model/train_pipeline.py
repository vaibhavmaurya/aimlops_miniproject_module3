import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from datasets import DataSet
import os
from pipeline import preprocess_pipeline

from config import DATASET_CONFIGURATION, MODEL_CONFIGURATION


current_file_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_file_path, 'trained_models')
metrics_path = os.path.join(current_file_path, 'trained_models', 'metrics.json')


print(f'''

    MODEL_CONFIGURATION: {MODEL_CONFIGURATION}

''')



# Define a dictionary to map the model names in the YAML to the actual scikit-learn classes
MODEL_NAME_MAP = {
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "SGDRegressor": SGDRegressor
}



def load_dataset():
    dataset = DataSet(DATASET_CONFIGURATION)
    return dataset



def train_and_save_models():
    """
    Function to train models based on configurations in YAML data,
    compute metrics, save models and metrics to disk.
    :param yaml_data: Dictionary object from parsed YAML data
    :return: metrics as a dictionary object
    """
    X_train, X_test, y_train, y_test = load_dataset().get_test_train_split()

    X_train_transformed = preprocess_pipeline.fit_transform(X_train)

    # Save the pipeline to a file
    joblib.dump(preprocess_pipeline, os.path.join(model_path, "preprocess_pipeline.joblib"))


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



def save_metrics(metrics):
    """
    Function to save metrics to a JSON file.
    :param metrics: metrics as a dictionary object
    """
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)



def main():
    """
    Main function
    """
    metrics = train_and_save_models()
    save_metrics(metrics)


if __name__ == "__main__":
    main()


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

