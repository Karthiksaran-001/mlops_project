from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.utils import load_object
import pandas as pd
import numpy as np


class CustomOHETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, oh_encoder_path):
        self.oh_encoder_path = oh_encoder_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        encoder = load_object(self.oh_encoder_path)
        ohe = encoder.transform(X[encoder.feature_names_in_])
        new_df = pd.DataFrame(ohe, columns=encoder.get_feature_names_out())
        new_df.reset_index(drop=True, inplace=True)
        return new_df
    
    def get_params(self, deep=True):
        return {"oh_encoder_path": self.oh_encoder_path}
    

class CustomOutlierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        X["cap_diameter"] = np.where(X["cap_diameter"] > 14.175 , 14.175 , np.where(X["cap_diameter"] < -3.225 , -3.225 , X["cap_diameter"] ))
        X["stem_height"] = np.where(X["stem_height"] > 11.83 , 11.83 , np.where(X["stem_height"] < 0.15 , 0.15 , X["stem_height"] ))
        X["stem_width"] = np.where(X["stem_width"] > 29.77 , 29.77 , np.where(X["stem_width"] < -10.45 , -10.45 , X["stem_width"] ))
        return X
    
