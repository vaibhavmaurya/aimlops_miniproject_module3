import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job


from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when
from pyspark.ml import Transformer
from pyspark.ml import Pipeline



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
        "paths": ["s3://vm-aimlops-2023/module_4/data/"],
        "recurse": True,
    },
    transformation_ctx="S3bucket_node1",
)


####################Imputers######################################

class YearMonthImputer(Transformer):
    """
    A custom Transformer which extracts year and month from a date column.
    """
    def __init__(self):
        pass

    def _transform(self, df):
        """
        Transforms the input DataFrame by adding year and month columns.
        Assumes that the input DataFrame has a column named 'dteday' of type Date.
        """
        # Convert 'dteday' to Date type if it's not already
        df = df.withColumn('dteday', F.to_date(col('dteday'), 'yyyy-MM-dd'))
        # Create 'yr' and 'mnth' columns based on 'dteday' column
        df = df.withColumn('yr', F.year(col('dteday'))) \
               .withColumn('mnth', F.month(col('dteday')))
        return df


class WeekdayImputer(Transformer):
    """
    A custom Transformer which imputes missing values in 'weekday' column with the weekday derived from 'dteday'.
    """
    def __init__(self):
        pass

    def _transform(self, df):
        """
        Transforms the input DataFrame by filling nulls in 'weekday' column with the weekday of 'dteday'.
        """
        # Impute missing 'weekday' values with the day of the week from 'dteday'
        df = df.withColumn('weekday', 
                           when(col('weekday').isNull(), F.date_format(col('dteday'), 'E')) \
                           .otherwise(col('weekday')))
        return df


class WeathersitImputer(Transformer):
    """
    A custom Transformer which imputes missing values in 'weathersit' column with 'Clear'.
    """
    def __init__(self):
        pass

    def _transform(self, df):
        """
        Transforms the input DataFrame by filling nulls in 'weathersit' column with 'Clear'.
        """
        # Impute missing 'weathersit' values with 'Clear'
        df = df.withColumn('weathersit', 
                           when(col('weathersit').isNull(), 'Clear') \
                           .otherwise(col('weathersit')))
        return df
        
######################Imputers end####################################

################ Applying imputers ###############################

# Convert DynamicFrame to Spark DataFrame for transformations
dataframe = S3bucket_node1.toDF()


# Create instances of our custom Transformers
year_month_imputer = YearMonthImputer()
weekday_imputer = WeekdayImputer()
weathersit_imputer = WeathersitImputer()

# Create a PySpark ML Pipeline with our Transformers
pipeline = Pipeline(stages=[year_month_imputer, weekday_imputer, weathersit_imputer])

# Apply the transformations
model = pipeline.fit(dataframe)
dataframe_transformed = model.transform(dataframe)

# Convert transformed DataFrame back to DynamicFrame
dynamic_frame_transformed = DynamicFrame.fromDF(dataframe_transformed, glueContext, "dynamic_frame_transformed")


################ Applying imputers ends ##########################

# Script generated for node ApplyMapping
# ApplyMapping_node2 = ApplyMapping.apply(
#     frame=S3bucket_node1,
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
#         ("casual", "int", "casual", "string"),
#         ("registered", "int", "registered", "string"),
#         ("cnt", "int", "cnt", "string"),
#     ],
#     transformation_ctx="ApplyMapping_node2",
# )


# Script generated for node S3 bucket
S3bucket_node3 = glueContext.write_dynamic_frame.from_options(
    frame=dynamic_frame_transformed,
    connection_type="s3",
    format="csv",
    connection_options={
        "path": "s3://vm-aimlops-2023/module_4/feature_store/",
        "partitionKeys": [],
    },
    transformation_ctx="S3bucket_node3",
)

job.commit()
