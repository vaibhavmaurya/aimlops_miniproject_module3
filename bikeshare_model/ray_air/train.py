import ray
from R_imputers import yearMonthColumnAdd, weekDayImputerfn, removeUnusedColumns
from R_mapper import CustomMapper
from ray.data.preprocessors import StandardScaler,LabelEncoder, OrdinalEncoder, Chain, SimpleImputer, BatchMapper, RobustScaler
import pyarrow.compute as pc
import pyarrow.csv as csv
from ray.train.xgboost import XGBoostTrainer, XGBoostPredictor
from ray.air.config import ScalingConfig
from ray import serve
from fastapi import Request
from ray.serve import PredictorDeployment
from ray.serve.http_adapters import json_request
import pandas as pd

PATH = '/Users/ankit8.agarwal/Downloads/'
FILE_NAME = 'bike-sharing-dataset.csv'
# Load data.
dataset = ray.data.read_csv(PATH + FILE_NAME, convert_options=csv.ConvertOptions(strings_can_be_null= True
, null_values = ['']))

# table = csv.read_csv(PATH + FILE_NAME, convert_options=csv.ConvertOptions(strings_can_be_null= True
# , null_values = ['']))
# df_new = table.to_pandas()
# data_panda  = dataset.to_pandas()
# print(df_new.info())
# print("Dataset" ,data_panda.info())
# print(df_new['weekday'].unique())
# print(pc.value_counts(table['weekday']))
# Split data into train and validation.
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.2)

# Create a test dataset by dropping the target column.
test_dataset = valid_dataset.drop_columns(cols=["cnt"])

yearMonthAdder = BatchMapper(yearMonthColumnAdd, batch_format="pandas")
weekDayImputer = BatchMapper(weekDayImputerfn, batch_format="pandas")
weatherSit = SimpleImputer(columns=["weathersit"], strategy="most_frequent")
ds = train_dataset.select_columns(cols=["weathersit"])
pandasdf = ds.to_pandas()
print(pandasdf['weathersit'].unique())
weather_mapping = ['Mist', 'Clear', 'Light Rain' ,'Heavy Rain'] 
#for i in ds.iter_rows():
    #if i['weathersit'] not in weather_mapping:
        #print ("Not present", i['weathersit'])
# yrEncoder = OrdinalEncoder(columns=["yr"])
outliers = RobustScaler(columns=['temp', 'atemp', 'hum', 'windspeed'])
mapper = CustomMapper()
print(train_dataset.columns())
removeColumns = BatchMapper(removeUnusedColumns, batch_format="pandas")

final_list = list(set(train_dataset.columns()) - set(['dteday', 'casual', 'registered', 'weekday']))
scaler = StandardScaler(final_list)

preprocessor = Chain(yearMonthAdder, weatherSit, weekDayImputer, mapper, outliers,removeColumns, scaler )

# dataset_transformed = preprocessor.fit_transform(train_dataset)
# valid_dataset = preprocessor.transform(valid_dataset)
# print(dataset_transformed.to_pandas().info())

trainer = XGBoostTrainer(
    # scaling_config=ScalingConfig(
    #     # Number of workers to use for data parallelism.
    #     num_workers=2,
    #     # Whether to use GPU acceleration.
    #     use_gpu=True,
    # ),
    label_column="cnt",
    num_boost_round=20,
    preprocessor = preprocessor,
    params={
        # XGBoost specific paramsparams={"objective": "reg:squarederror"},
        "objective": "reg:squarederror",
        # "tree_method": "gpu_hist",  # uncomment this to use GPU for training
        "eval_metric": ["mae", "rmse"],
    },
    datasets={"train": train_dataset, "valid": valid_dataset},
)
result = trainer.fit()
print(result.metrics)
#print(dataset.show)

async def adapter(request: Request):
    content = await request.json()
    print(content)
    return pd.DataFrame.from_dict(content)


serve.run(
    PredictorDeployment.options(name="XGBoostService").bind(
        XGBoostPredictor, result.checkpoint, batching_params=False, http_adapter=adapter
    )
)