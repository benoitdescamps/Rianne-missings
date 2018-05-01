from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer
import numpy as np

class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """Transformer which augments the data given a pandas dataframe and output another pandas dataframe with possibly more columns.
    Attributes:
        None
    """
    def __init__(self):
        self.columns = columns # type: List[str]

    def fit(self, X, y=None):
        #add assert X is pd.DataFrame
        return self

    def transform(self, X):
        return X   
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Transformers which selects  a list of given columns from a pandas DataFrame.
    Attributes:
        columns (:obj:`list` of :obj:`str`): list of columns being selected from a pandas DataFrame.
    """
    def __init__(self, columns):
        self.columns = columns # type: List[str]

    def fit(self, X, y=None):
        #add assert X is pd.DataFrame
        return self

    def transform(self, X):
        return X[list(self.columns)]
    
class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    """Transformers which selects  a list of given columns from a pandas DataFrame.
    Attributes:
        columns (:obj:`list` of :obj:`list` of obj:`str`): list of  list of columns being selected from a pandas DataFrame and used for the different transformers.
    """
    def __init__(self, columns):
        self.columns = columns # type: List[str]
        self.transformer = FeatureUnion(
            [                
                ('mean_imputations', Pipeline([
                                ('selector', ColumnSelector(columns=None))
                               ,('imputer', Imputer(strategy='mean'))                            ])),
                ('median_imputations', Pipeline([
                                ('selector', ColumnSelector(columns=None))
                               ,('imputer', Imputer(strategy='median'))                            ])),
                ('no_imputations', Pipeline([
                                ('selector', ColumnSelector(columns=None))        ]))
            ], 
        )
       

    def fit(self, X, y=None):
        #add assert X is pd.DataFrame
        self.transformer.set_params(**{
            'mean_imputations__selector__columns':list(self.columns[0]),
            'median_imputations__selector__columns':list(self.columns[1]),
            'no_imputations__selector__columns':list(self.columns[2])
        })
        self.transformer.fit(X,y)
        return self

    def transform(self, X):
        return self.transformer.transform(X)
