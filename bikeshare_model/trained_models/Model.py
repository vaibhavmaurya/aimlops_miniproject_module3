__all__ = ["load_models_and_predict", "get_metrics"]

import joblib
import json
import os



def load_models_and_predict(input_data, config):
    """
    Function to load saved models, make predictions on input data.

    Args:
        yaml_data (dict): Dictionary object from parsed YAML data.
        input_data (array-like): Input data to make predictions on.

    Returns:
        dict: A dictionary with the input data and the predictions from each model.
    """
    print('''You want to predict the following data:''')
    print()
    print(input_data)
    preprocess_pipeline = joblib.load(os.path.join(config["pipeline_save_path"], "preprocess_pipeline.joblib"))
    DATASET_CONFIGURATION = config.get("dataset_configuration", None)
    MODEL_CONFIGURATION = config.get("models", None)
    model_path = config.get("model_save_path", None)

    output = {}

    input_features = list(input_data.columns)
    input_features.remove(DATASET_CONFIGURATION["Target_feature"])
    test_data_input = input_data[input_features]

    # Transform the input data using the saved pipeline
    test_data_input = preprocess_pipeline.transform(test_data_input)

    # Load the models and make predictions
    for model_info in MODEL_CONFIGURATION:
        model_name = model_info["model_name"]

        # Load the model from a file
        model = joblib.load(os.path.join(model_path, f"{model_name}.joblib"))
        # Make predictions on the input data
        y_pred = model.predict(test_data_input)

        # Add the predictions to the output
        output[model_name] = y_pred.tolist()

    return output



def get_metrics(config):
    """
    Function to load metrics from a JSON file.

    Returns:
        dict: A dictionary with the metrics.
    """
    model_metrics_save_path = config.get("model_metrics_save_path", None)

    with open(model_metrics_save_path, "r") as f:
        metrics = json.load(f)

    return metrics
