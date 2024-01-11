import os
import sys
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customException
from dataclasses import dataclass
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.utils.utils import save_object , load_object


@dataclass
class DataTransformationConfig:
    encoder_path:str=os.path.join("artifacts\Model","OneHotEncoder.pickle")


class DataTransformation:
    def __init__(self):
        self.transform_config = DataTransformationConfig()

    def outlier_transform(self , df):
        logging.info("\t\t\t Outlier Capping for {}\n".format(df.columns))
        df["cap-diameter"] = np.where(df["cap-diameter"] > 14.175 , 14.175 , np.where(df["cap-diameter"] < -3.225 , -3.225 , df["cap-diameter"] ))
        df["stem-height"] = np.where(df["stem-height"] > 11.83 , 11.83 , np.where(df["stem-height"] < 0.15 , 0.15 , df["stem-height"] ))
        df["stem-width"] = np.where(df["stem-width"] > 29.77 , 29.77 , np.where(df["stem-width"] < -10.45 , -10.45 , df["stem-width"] ))
        return df
    
    def transform_ohe(self,df: pd.DataFrame):
        encoder = load_object(self.transform_config.encoder_path)
        ohe = encoder.transform(df[encoder.feature_names_in_])
        new_df = pd.DataFrame(ohe , columns = encoder.get_feature_names_out())
        df.reset_index(drop=True, inplace=True)
        new_df.reset_index(drop=True, inplace=True)
        df.drop(columns=encoder.feature_names_in_ , axis = 1 , inplace= True)
        df = pd.concat([df , new_df] , axis = 1)
        return df


    
    def get_data_transformation():
        logging.info('Data Transformation initiated')
        try:
            numerical_cols = ['cap-diameter', 'stem-height','stem-width']
            categorical_cols = ['cap-shape', 'cap-surface', 'cap-color','does-bruise-or-bleed', 'gill-attachment', 'gill-color', 'stem-color', 'ring-type', 'habitat', 'season']
            encoding_cols = ['does-bruise-or-bleed', 'season']

            # Define the custom ranking for each ordinal variable
            cap_shape = ['o', 's', 'c', 'p', 'f', 'x', 'b']
            cap_surface = ['s', 'g', 'w', 'd', 'y', 'h', 'k', 'e', 't', 'i', 'l']
            cap_color = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            gill_attachment = ['s', 'd', 'p', 'f', 'x', 'e', 'a']
            gill_color = ['o', 'g', 'r', 'p', 'f', 'y', 'n', 'b', 'u', 'e', 'w', 'k']
            stem_color = ['b', 'e', 'f', 'g', 'k', 'l', 'n', 'o', 'p', 'r', 'u', 'w', 'y']
            ring_type =  ['g', 'r', 'z', 'f', 'p', 'm', 'e', 'l']
            habitat = ['d', 'g', 'h', 'l', 'm', 'p', 'u', 'w']

            

        except Exception as e:
            logging.info()
            raise customException(e , sys)




    def initate_data_transformation(self, train_path , test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        except Exception as e:
            logging.error()
            raise customException(e , sys)
