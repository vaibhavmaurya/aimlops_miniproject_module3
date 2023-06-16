__all__ = ['get_pipeline']

from datasets import DataSet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bikeshare_model.processing import WeekdayImputer,        \
                       YearMonthImputer,      \
                       WeathersitImputer,     \
                       Mapper,                \
                       OutlierHandler,        \
                       WeekdayOneHotEncoder,  \
                       RemoveUnusedColumns




def get_pipeline(dataset_configuration: dict) -> Pipeline:
    """
    Function to create a pipeline based on configurations in YAML data
    :param yaml_data: Dictionary object from parsed YAML data
    :return: pipeline object
    """ 

    # print(dataset_configuration)


    # dataset = DataSet(dataset_configuration) \
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
    outlier_handler = OutlierHandler(columns=dataset_configuration['numerical_variables'])

    # Create the one hot encoder
    weekday_onehot_encoder = WeekdayOneHotEncoder(column='weekday')

    # Remove unused columns
    remove_unused_columns = RemoveUnusedColumns(unused_columns=dataset_configuration['unused_column'])

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
    return preprocess_pipeline



