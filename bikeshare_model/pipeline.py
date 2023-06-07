__all__ = ['preprocess_pipeline']

from config import DATASET_CONFIGURATION
from datasets import DataSet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from processing import WeekdayImputer,        \
                       YearMonthImputer,      \
                       WeathersitImputer,     \
                       Mapper,                \
                       OutlierHandler,        \
                       WeekdayOneHotEncoder,  \
                       RemoveUnusedColumns



# print(DATASET_CONFIGURATION)


# dataset = DataSet(DATASET_CONFIGURATION) \
#             .extract_year_month()

# print(dataset.head())


# Year and month imputer
year_month_imputer = YearMonthImputer()

# Create the imputers
weekday_imputer = WeekdayImputer()

weathersit_imputer = WeathersitImputer()

# Create the mapper
mapper = Mapper()

# Create the outlier handler
outlier_handler = OutlierHandler(columns=DATASET_CONFIGURATION['numerical_variables'])

# Create the one hot encoder
weekday_onehot_encoder = WeekdayOneHotEncoder(column='weekday')

# Remove unused columns
remove_unused_columns = RemoveUnusedColumns(unused_columns=DATASET_CONFIGURATION['unused_column'])

# Standard Scaler
scaler = StandardScaler()


# Create the pipeline
preprocess_pipeline = Pipeline([
    ('year and month imputer', year_month_imputer),
    ('weekday imputer', weekday_imputer),
    ('weathersit imputer', weathersit_imputer),
    ('outlier_handler', outlier_handler),
    ('weekday one hot encoder', weekday_onehot_encoder),
    ('mapper', mapper),
    ('Remove unused columns', remove_unused_columns),
    ('Standard Scaling', scaler),
])

# Fit the pipeline
# pipeline.fit(dataset)
# dataset = pipeline.transform(dataset)

# print(dataset[0, :])



