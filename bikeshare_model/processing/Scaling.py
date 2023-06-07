__all__ = ["WeekdayOneHotEncoder"]

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, column=None):
        """Initialize the encoder with the name of the column to transform.
        
        Parameters
        ----------
        column : str
            The name of the column to transform.
        """
        self.column = column
        self.encoder = OneHotEncoder(sparse=False)

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder using the given DataFrame.
        
        Parameters
        ----------
        X : pandas DataFrame
            The input data to fit the encoder.
        
        Returns
        -------
        self
        """
        self.encoder.fit(X[[self.column]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the specified column of the given DataFrame.
        
        Parameters
        ----------
        X : pandas DataFrame
            The input data to transform.
        
        Returns
        -------
        X : pandas DataFrame
            The transformed input data.
        """
        # Transform the specified column
        transformed_column = self.encoder.transform(X[[self.column]])

        # Create DataFrame from transformed column
        transformed_df = pd.DataFrame(
            transformed_column, 
            columns=[f"{self.column}_{day}" for day in self.encoder.categories_[0]], 
            index=X.index)

        # Drop the original column from the input DataFrame and add the transformed columns
        X_transformed = pd.concat([X.drop(self.column, axis=1), transformed_df], axis=1)

        return X_transformed
