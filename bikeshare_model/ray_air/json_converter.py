import json


PATH = '/Users/ankit8.agarwal/LearningSparkV2/chapter2/py/src/data/'
FILE_NAME = 'prod_mart_master_1.json'
FILE_NAME_OUTPUT = 'prod_mart_master_output_1.json'
input_file = PATH + FILE_NAME  # Replace with the path to your input JSON file
output_file = PATH + FILE_NAME_OUTPUT  # Replace with the path for the output file

# Read the input JSON file
with open(input_file, 'r') as file:
    data = json.load(file)

# Write each JSON object on a separate line in the output file
with open(output_file, 'w') as file:
    for obj in data:
        json.dump(obj, file)
        file.write('\n')