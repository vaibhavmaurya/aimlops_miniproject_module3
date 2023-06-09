from flask import Flask, request, jsonify
import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bikeshare_model import get_metrics, final_prediction, get_config


app = Flask(__name__)

# Assuming you have models loaded, for example:
# from sklearn.externals import joblib
# LinearRegression = joblib.load('LinearRegression.pkl')
# RandomForestRegressor = joblib.load('RandomForestRegressor.pkl')
# SGDRegressor = joblib.load('SGDRegressor.pkl')

CONFIG = get_config("/mnt/c/Users/91961/Documents/Learn/AIML/MLOps/MiniProject/Module3/config/config.yml")


@app.route('/predict', methods=['POST'])
def predict_api():
    """
    This function takes in a POST request with JSON data containing the 
    input for the models. The function applies the models to the data 
    and returns a JSON response with the input data and the predictions.

    Returns:
    - 400 status if no data is provided.
    - A JSON response containing the input data and the model predictions.
    """
    data = request.json

    # If data is not provided, return bad request status.
    if not data:
        return jsonify({'message': 'No input data provided'}), 400

    # Assume we have a function `predict` which uses the models to predict output
    print("Start Predicting")
    predictions = final_prediction([data], CONFIG)
    print("Predicting Done")

    response = {
        'input': data,
        'predictions': predictions
    }

    return jsonify(predictions)



@app.route('/metrics', methods=['GET'])
def get_model_performance():
    """
    This function takes in a GET request and returns a JSON response 
    containing the contents of the 'metrics.json' file.

    Returns:
    - 404 status if the file is not found.
    - A JSON response containing the contents of the 'metrics.json' file.
    """
    try:
        # Return the JSON data as a response
        return jsonify(get_metrics(CONFIG))
    
    except FileNotFoundError:
        # Return a 404 status if the file is not found
        return jsonify({'message': 'metrics.json not found'}), 404


if __name__ == '__main__':
    app.run(debug=True)
