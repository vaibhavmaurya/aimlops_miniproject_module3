# Bike Data Machine Learning Pipeline

## Introduction
This project aims to perform a complete pipeline of data preprocessing on bike data and train three models: Linear Regressor, SGD Regressor, and Random Forest Regressor. The project can accommodate additional models based on the configurations specified in a YAML configuration file.

## Project Structure
The project consists of the following components:

1. `datasets` folder: Contains the dataset used for training the models.
2. `trained_models` directory: Stores the trained models along with a `metrics.json` file that contains the R2 square and MSE metrics for each model.
3. `train_pipeline.py`: Python script used to train the models via the command line.
4. `pipeline.py`: Module containing the pipeline for data preprocessing steps.
5. `service.py`: Flask REST API implementation with two endpoints:
   - `/predict` (POST): Accepts a JSON input and provides predictions from the trained models.
   - `/metrics` (GET): Returns the performance metrics of the models on the test data, retrieved from the `metrics.json` file.

## Usage

### Training the Models
To train the models, follow these steps:

1. Ensure that the dataset is placed in the `datasets` folder.
2. Open the command line interface.
3. Run the following command:

`> python train_pipeline.py`

### Running the Flask REST API
To run the Flask REST API, perform the following steps:

1. Open the command line interface.
2. Run the following command:

`> python service.py`


## Conclusion
This project provides a comprehensive pipeline for data preprocessing and model training on bike data. It offers flexibility for incorporating different models based on the configurations specified in the YAML file. Additionally, the Flask REST API endpoints allow for easy prediction and access to model performance metrics.

