from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, month, year, lit, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
# Create a SparkSession
PATH = '/Users/ankit8.agarwal/Downloads/'
FILE_NAME = 'bike-sharing-dataset.csv'
spark = SparkSession.builder \
    .appName("CSV to DataFrame") \
    .getOrCreate()

# Read the CSV file into a DataFrame
df = spark.read.csv(PATH + FILE_NAME, header=True, inferSchema=True)

# Display the DataFrame


column_name = "weekday"

# Count the number of null values in the specified column
null_count = df.filter(col(column_name).isNull()).count()
print(f"Number of null values in {column_name}: {null_count}")

#noNullsDF = df.na.fill(col("dteday"), column_name)
#couldn't find any inbuilt function to replace the value from other column, created a new column for weekday

df = df.withColumn("day_of_week", dayofweek(col("dteday")))
df = df.withColumn("yr", year(col("dteday")))
df = df.withColumn("mnth", month(col("dteday")))

column_types = df.dtypes

# Separate numerical and categorical columns
numerical_columns = []
categorical_columns = []

for column, dtype in column_types:
    if dtype in ("int", "bigint", "float", "double"):
        numerical_columns.append(column)
    else:
        categorical_columns.append(column)

# Print the results
print("Numerical Columns:", numerical_columns)
print("Categorical Columns:", categorical_columns)

trainDF, testDF = df.randomSplit([.8, .2], seed=42)


grouped = trainDF.groupBy("weathersit").count()

# Sort by count in descending order
sorted_grouped = grouped.sort(col("count").desc())

# Get the most frequent value
most_frequent_value = sorted_grouped.first()[0]

print("most frequent in weathersit", most_frequent_value)
null_count = trainDF.filter(col('weathersit').isNull()).count()
#print("null count", null_count)
trainDF = trainDF.na.fill(most_frequent_value, 'weathersit')
null_count = trainDF.filter(col('weathersit').isNull()).count()
print("null count", null_count)


numerical_columns = ['temp', 'atemp' , 'windspeed' , 'hum']
#Remove Outliers
# import matplotlib.pyplot as plt
# pandas_df = trainDF.toPandas()
# pandas_df[numerical_columns].boxplot()
# plt.xticks(rotation= 60)
# plt.show()



def remove_outliers(trainDF ,numerical_column):

    lower_quartile = trainDF.approxQuantile(numerical_column, [0.25],0)[0]
    upper_quartile = trainDF.approxQuantile(numerical_column, [0.75],0)[0]
    # Calculate IQR
    iqr = upper_quartile - lower_quartile

    # Define lower and upper bounds
    lower_bound = lower_quartile - 1.5 * iqr
    upper_bound = upper_quartile + 1.5 * iqr
    print("lower",lower_bound, numerical_column)
    print("upper",upper_bound, numerical_column)
    # Replace values outside IQR range with null
    trainDF = trainDF.withColumn(numerical_column, 
                                        when(trainDF[numerical_column] < lower_bound, 1.111).when(trainDF[numerical_column] > upper_bound, 
                                           upper_bound).otherwise(trainDF[numerical_column]))
    
    return trainDF

for col in numerical_columns:
    trainDF = remove_outliers(trainDF, col)

# trainDF.show()
# import matplotlib.pyplot as plt
# pandas_df = trainDF.toPandas()
# pandas_df[numerical_columns].boxplot()
# plt.xticks(rotation= 60)
# plt.show()

mapping_columns = ['yr' , 'mnth', 'season' ,  'weathersit' , 'holiday' , 'workingday' , 'hr' ]

#Mapper

def createMapping(df, column_name):
    stringIndexer = StringIndexer(inputCol= column_name, outputCol = column_name + "_index" )
    model = stringIndexer.fit(df)
    df = model.transform(df)
    return df

for col in mapping_columns:
    trainDF = createMapping(trainDF, col)

trainDF.show()


#Pipeline

mapping_columns = ['yr' , 'mnth', 'season' ,  'weathersit' , 'holiday' , 'workingday' , 'hr']
ohe_columns = ['day_of_week']
scaler_colums = ['temp' , 'atemp' , 'hum' , 'windspeed']
indexOutputCols = [x + "_Index" for x in mapping_columns] 
oheOutputCols = [x + "_OHE" for x in ohe_columns]
scalerOutputCols =  [x + "_SCL" for x in scaler_colums]
stringIndexer = StringIndexer(inputCols=mapping_columns,
                                  outputCols=indexOutputCols,
                                  handleInvalid="skip")
oheEncoder = OneHotEncoder(inputCols=ohe_columns,
                               outputCols=oheOutputCols)

#sclaerAssembler = VectorAssembler(inputCols=scaler_colums, outputCol="features_Scaled")
#trainDF = assembler.transform(trainDF)




inputColums = indexOutputCols + oheOutputCols + scaler_colums

vecAssembler = VectorAssembler(inputCols=inputColums, outputCol="features")
scaler = StandardScaler( inputCol="features", outputCol="scaledFeatures")

rfr = RandomForestRegressor(labelCol="cnt", featuresCol="scaledFeatures")
pipeline = Pipeline(stages = [stringIndexer, oheEncoder, vecAssembler, scaler, rfr])


#MLFLOW

import mlflow
import mlflow.spark
import pandas as pd

with mlflow.start_run(run_name="random-forest") as run: # Log params: num_trees and max_depth 
    mlflow.log_param("num_trees", rfr.getNumTrees()) 
    mlflow.log_param("max_depth", rfr.getMaxDepth())

# Log model
    pipelineModel = pipeline.fit(trainDF)
    
    mlflow.spark.log_model(pipelineModel, "model")
    predDF = pipelineModel.transform(testDF)
    predDF.select("scaledFeatures", "cnt", "prediction").show(5)

    regressionEvaluator = RegressionEvaluator(
        predictionCol="prediction",
        labelCol="cnt",
        metricName="rmse")
    rmse = regressionEvaluator.evaluate(predDF) 
    print(f"RMSE is {rmse:.1f}")
    r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF) 
    print(f"R2 is {r2}")
    mlflow.log_metrics({"rmse": rmse, "r2": r2})

    rfModel = pipelineModel.stages[-1]
    pandasDF = (pd.DataFrame(list(zip(vecAssembler.getInputCols(),
                                        rfModel.featureImportances)),
                               columns=["feature", "importance"])
                  .sort_values(by="importance", ascending=False))
    # First write to local filesystem, then tell MLflow where to find that file
    pandasDF.to_csv("feature-importance.csv", index=False)
    mlflow.log_artifact("feature-importance.csv")
