import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

import boto3
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when
from pyspark.ml import Transformer
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql.functions import create_map, lit, col
from itertools import chain
from pyspark.ml.feature import VectorSlicer
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.linalg import DenseVector, VectorUDT



args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Script generated for node S3 bucket
S3bucket_node1 = glueContext.create_dynamic_frame.from_options(
    format_options={
        "quoteChar": '"',
        "withHeader": True,
        "separator": ",",
        "multiline": True,
        "optimizePerformance": False,
    },
    connection_type="s3",
    format="csv",
    connection_options={
        "paths": ["s3://vm-aimlops-2023/module_4/data/training/"],
        "recurse": True,
    },
    transformation_ctx="S3bucket_node1",
)


####################Imputers######################################
        

class OutlierHandler(Transformer):
    """
    A custom Transformer which detects outliers in specified columns using the Interquartile Range (IQR) method,
    and replaces them with the lower or upper bound respectively.
    """
    def __init__(self, columns=None, factor=1.5):
        self.columns = columns
        self.factor = factor

    def _transform(self, df):
        """
        Transforms the input DataFrame by replacing outlier values.
        """
        for column in self.columns:
            quantiles = df.stat.approxQuantile(column, [0.25, 0.75], 0.05)
            IQR = quantiles[1] - quantiles[0]
            lower_bound = quantiles[0] - self.factor * IQR
            upper_bound = quantiles[1] + self.factor * IQR
            df = df.withColumn(column, 
                               when(col(column) < lower_bound, lower_bound) \
                               .when(col(column) > upper_bound, upper_bound) \
                               .otherwise(col(column)))
        return df

        

        
######################Imputers end####################################






################ Applying imputers ###############################

# Convert DynamicFrame to Spark DataFrame for transformations
dataframe = S3bucket_node1.toDF()

# Lets do some transformations here
# Specify the columns to be casted
int_columns = ["casual", "registered", "cnt"]  # replace with your actual column names
decimal_columns = ["temp", "atemp", "hum", "windspeed"]  # replace with your actual column names

# Cast to integer
for col_name in int_columns:
    dataframe = dataframe.withColumn(col_name, col(col_name).cast("integer"))

# Cast to decimal
for col_name in decimal_columns:
    dataframe = dataframe.withColumn(col_name, col(col_name).cast("decimal"))
    
    
    
# Change weathersit column
dataframe = dataframe.withColumn('weathersit', when(col('weathersit').isNull(), 'Clear').otherwise(col('weathersit')))


# Convert 'dteday' to Date type if it's not already
dataframe = dataframe.withColumn('dteday', F.to_date(col('dteday'), 'yyyy-MM-dd'))
# Create 'yr' and 'mnth' columns based on 'dteday' column
dataframe = dataframe.withColumn('yr', F.year(col('dteday'))) \
       .withColumn('mnth', F.month(col('dteday')))

# Convert weekday column
dataframe = dataframe.withColumn('weekday', 
                           when(col('weekday').isNull(), F.date_format(col('dteday'), 'E')) \
                           .otherwise(col('weekday')))
                           
# Impute the weekday to One hot encoding
# List of three-letter abbreviations for the days of the week
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Create a column for each day of the week
for day in days:
    dataframe = dataframe.withColumn("weekday_"+day, (col("weekday") == day).cast("int"))
    
# Drop the original 'weekday' column
dataframe = dataframe.drop("weekday")

# Now, the specified columns in dataframe have been casted to the specified types

# Create instances of our custom Transformers

# Create the outlier handler
outlier_handler = OutlierHandler(columns=['temp', 'atemp', 'hum', 'windspeed'])


# Following is the mapping
from pyspark.sql.functions import create_map, lit, col
from itertools import chain

# Mapping dictionaries
yr_mapping = {2011: 0, 2012: 1}
mnth_mapping = {'January': 0, 'February': 1, 'December': 2, 
                'March': 3, 'November': 4, 'April': 5, 
                'October': 6, 'May': 7, 'September': 8, 
                'June': 9, 'July': 10, 'August': 11}
season_mapping = {'spring': 0, 'winter': 1, 'summer': 2, 'fall': 3}
weather_mapping = {'Heavy Rain': 0, 'Light Rain': 1, 'Mist': 2, 'Clear': 3}
workingday_mapping = {'No': 0, 'Yes': 1}
hour_mapping = {'4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, 
                '12am': 5, '6am': 6, '11pm': 7, '10pm': 8, 
                '10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, 
                '8pm': 14, '2pm': 15, '1pm': 16, 
                '12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, 
                '6pm': 22, '5pm': 23}
holiday_mapping = {'Yes': 0, 'No': 1}

mapping = {
    'yr'        :yr_mapping,
    'mnth'      :mnth_mapping,
    'season'    :season_mapping,
    'weathersit':weather_mapping,
    'holiday'   :holiday_mapping,
    'workingday':workingday_mapping,
    'hr'        :hour_mapping
}

# Apply the mappings to the DataFrame
for column, map_dict in mapping.items():
    mapping_expr = create_map([lit(x) for x in chain(*map_dict.items())])
    dataframe = dataframe.withColumn(column, mapping_expr.getItem(col(column)))
    
    
    
# Remove unused columns here
# List of columns to remove
columns_to_remove = ['dteday', 'casual', 'registered']

# Remove the columns
dataframe = dataframe.drop(*columns_to_remove)


# First we need to convert the input data frame to a dense vector 
# VectorAssembler is used to transform and return a new DataFrame with all of the feature columns in a vector column.
# Specify the features to be scaled
features_to_scale = ['temp', 'atemp', 'hum', 'windspeed']  # replace with your actual column names


# # # First, assemble the features to be scaled into a single vector column
assembler = VectorAssembler(inputCols=features_to_scale, outputCol="assembled")
scaler = MinMaxScaler(inputCol="assembled", outputCol="scaled")

# # # Create a PySpark ML Pipeline with our Transformers
pipeline = Pipeline(stages=[assembler, scaler])

# # Define a pipeline for transformation
pipeline = Pipeline(stages=[assembler, scaler])

# # Fit and transform
model = pipeline.fit(dataframe)
df_scaled = model.transform(dataframe)

# # Save this pipeline
# local_path = "/tmp/standardscaler"
# model.write().overwrite().save(local_path)
# # model.write().overwrite().save("s3://vm-aimlops-2023/module_4/feature_store/standardscaler")

# # Use Boto3 to copy the local file to S3
# s3 = boto3.resource('s3')
# # Get a handle to the S3 bucket
# s3.Bucket("vm-aimlops-2023").upload_file(local_path, "s3://vm-aimlops-2023/module_4/feature_store/standardscaler")

# # Convert the vector to an array
to_array = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))
df_scaled = df_scaled.withColumn("scaled_array", to_array(df_scaled["scaled"]))

# # Now we need to disassemble the scaled features back to their original form
for i, feature in enumerate(features_to_scale):
    df_scaled = df_scaled.withColumn(feature, df_scaled["scaled_array"].getItem(i))

# # You can drop the temporary columns if you want
df_scaled = df_scaled.drop("assembled", "scaled", "scaled_array")


# # Convert transformed DataFrame back to DynamicFrame
dynamic_frame_transformed = DynamicFrame.fromDF(df_scaled, glueContext, "dynamic_frame_transformed")


################ Applying imputers ends ##########################



# # Script generated for node ApplyMapping
# ApplyMapping_node2 = ApplyMapping.apply(
#     frame=dynamic_frame_transformed,
#     mappings=[
#         ("dteday", "string", "dteday", "string"),
#         ("season", "string", "season", "string"),
#         ("hr", "string", "hr", "string"),
#         ("holiday", "string", "holiday", "string"),
#         ("weekday", "string", "weekday", "string"),
#         ("workingday", "string", "workingday", "string"),
#         ("weathersit", "string", "weathersit", "string"),
#         ("temp", "decimal", "temp", "string"),
#         ("atemp", "decimal", "atemp", "string"),
#         ("hum", "decimal", "hum", "string"),
#         ("windspeed", "decimal", "windspeed", "string"),
#         ("casual", "decimal", "casual", "string"),
#         ("registered", "decimal", "registered", "string"),
#         ("cnt", "decimal", "cnt", "string"),
#     ],
#     transformation_ctx="ApplyMapping_node2",
# )


glueContext.purge_s3_path("s3://vm-aimlops-2023/module_4/feature_store/training/", options={}, transformation_ctx="")



# Script generated for node S3 bucket
S3bucket_node3 = glueContext.write_dynamic_frame.from_options(
    frame=dynamic_frame_transformed,
    connection_type="s3",
    format="csv",
    connection_options={
        "path": "s3://vm-aimlops-2023/module_4/feature_store/training/",
        "partitionKeys": [],
    },
    transformation_ctx="S3bucket_node3",
)

job.commit()
