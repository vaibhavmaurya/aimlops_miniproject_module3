import ray
from R_imputers import yearMonthColumnAdd, weekDayImputerfn
from R_mapper import CustomMapper
from ray.data.preprocessors import StandardScaler,LabelEncoder, OrdinalEncoder, Chain, SimpleImputer, BatchMapper, RobustScaler
import pyarrow.compute as pc
import pyarrow.csv as csv
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
print(train_dataset.to_pandas().info())
scaler = StandardScaler()
preprocessor = Chain(yearMonthAdder, weatherSit, weekDayImputer, mapper, outliers,scaler )


dataset_transformed = preprocessor.fit_transform(train_dataset)

print(dataset_transformed.to_pandas().info())
#print(dataset.show)