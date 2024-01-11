import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.utils.utils import save_object , load_object 
from src.utils.custom_transformation import CustomOHETransformer , CustomOutlierTransformer

@dataclass
class DataTransformationConfig:
    oh_encoder_path:str=os.path.join("artifacts\Model","OneHotEncoder.pickle")
    preprocessor_obj_file_path=os.path.join('artifacts\Model','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation(self):
        logging.info('Data Transformation initiated')
        try:
            numerical_cols = ['cap_diameter', 'stem_height','stem_width']
            categorical_cols = ['cap_shape', 'cap_surface', 'cap_color', 'gill_attachment', 'gill_color', 'stem_color', 'ring_type', 'habitat']
            pass_through = ["does-bruise-or-bleed" , "season"]

            cap_shape = ['o', 's', 'c', 'p', 'f', 'x', 'b']
            cap_surface = ['s', 'g', 'w', 'd', 'y', 'h', 'k', 'e', 't', 'i', 'l']
            cap_color = ['o', 'g', 'r', 'p', 'y', 'n', 'u', 'b', 'e', 'w', 'k', 'l']
            gill_attachment = ['s', 'd', 'p', 'f', 'x', 'e', 'a']
            gill_color = ['o', 'g', 'r', 'p', 'f', 'y', 'n', 'b', 'u', 'e', 'w', 'k']
            stem_color = ['b', 'e', 'f', 'g', 'k', 'l', 'n', 'o', 'p', 'r', 'u', 'w', 'y']
            ring_type =  ['g', 'r', 'z', 'f', 'p', 'm', 'e', 'l' , 'c']
            habitat = ['d', 'g', 'h', 'l', 'm', 'p', 'u', 'w']

            num_pipeline=Pipeline(
                steps=[
                    ('custom_outlier' , CustomOutlierTransformer()),
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())] )

            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cap_shape,cap_surface,cap_color,gill_attachment,gill_color,stem_color,ring_type,habitat] , dtype=int))])
            
            custom_ohe = CustomOHETransformer(oh_encoder_path= self.data_transformation_config.oh_encoder_path)

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols),
            ('custom_ohe', custom_ohe, pass_through)
            ])
            
            logging.info('\t\t\t Finish the PreProcessing \n')
            return preprocessor
        except Exception as e:
            logging.error("\t\t\t (DataIngestion) :"+str(CustomException(e , sys))+ "\n")
            raise CustomException(e , sys)




    def initate_data_transformation(self, train_path , test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("\t\t\t read train and test data complete\n")
            logging.info(f'\t\t\t Train Dataframe Head : \n{train_df.head().to_string()}\n')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}\n')
            logging.info('\t\t\t Finish Outlier Capping and One hot Encoder \n')
            preprocessing_obj = self.get_data_transformation()
            target_column_name = 'class'

            input_feature_train_df = train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("\t\t\t Input need for the model is : {}\n".format(input_feature_train_df.head(2).to_string()))

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            


            logging.info("\t\t\t Applying preprocessing object on training and testing datasets\n")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df , dtype=int)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df , dtype=int)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
        
            logging.info("\t\t\t preprocessing pickle file saved\n")            
            return (train_arr,test_arr)
        except Exception as e:
            logging.error("\t\t\t (DataIngestion) :"+str(CustomException(e , sys))+ "\n")
            raise CustomException(e , sys)
            



    