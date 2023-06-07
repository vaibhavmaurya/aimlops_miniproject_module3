__all__ = ['OutlierHandler', 'RemoveUnusedColumns']

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Impute outliers in pandas DataFrame using Interquartile Range (IQR).
    
    This imputer treats values below Q1 - factor * IQR or above Q3 + factor * IQR as outliers,
    and replaces them with the lower or upper bound respectively.
    
    Parameters:
    -----------
    columns : list
        List of column names to apply imputation to. If None, imputation is applied to all numerical columns.
    
    factor : float
        The multiplier for IQR. Defines the step of the outlier bounds. Default is 1.5.
    
    Attributes:
    -----------
    lower_bounds_ : dict
        Lower bound for outlier detection for each column.
        
    upper_bounds_ : dict
        Upper bound for outlier detection for each column.
        
    outliers_ : dict
        Outliers detected in each column during fitting.
    """
    def __init__(self, columns=None, factor=1.5):
        self.columns = columns
        self.factor = factor

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the imputer according to the IQR of X.
        
        Parameters:
        -----------
        X : pandas DataFrame, shape (n_samples, n_features)
            The data used to compute the median and the IQR for each feature.
            
        y : Ignored
            Not used, present for API consistency by convention.
            
        Returns:
        --------
        self
        """
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.outliers_ = {}
        for column in self.columns:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            # Detect outliers
            outliers = X[(X[column] < lower_bound) | (X[column] > upper_bound)][column]
            if len(outliers) > 0:
                self.lower_bounds_[column] = lower_bound
                self.upper_bounds_[column] = upper_bound
                self.outliers_[column] = outliers
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute all outliers in X.
        
        Parameters:
        -----------
        X : pandas DataFrame, shape (n_samples, n_features)
            The input data to complete.
            
        Returns:
        --------
        X : pandas DataFrame, shape (n_samples, n_features)
            The imputed input data.
        """
        for column in self.outliers_.keys():
            X[column] = np.where((X[column] < self.lower_bounds_[column]), self.lower_bounds_[column], X[column])
            X[column] = np.where((X[column] > self.upper_bounds_[column]), self.upper_bounds_[column], X[column])
        return X

    def get_outliers(self):
        """Get the outliers detected during fitting.
        
        Returns:
        --------
        outliers : dict
            The outliers for each feature.
        """
        return self.outliers_
        


class RemoveUnusedColumns(BaseEstimator, TransformerMixin):
    """ Remove unused columns """

    def __init__(self, unused_columns:list):
        self.weathersit = None
        self.unused_columns = unused_columns

    def fit(self, X: pd.DataFrame, y=None):
        """ Learn existing unused columns """
        self.unused_columns = [col for col in self.unused_columns if col in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Remove unused columns """
        X.drop(columns=self.unused_columns, inplace=True)
        return X 