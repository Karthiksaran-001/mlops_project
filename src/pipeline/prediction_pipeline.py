import os
import sys
import pandas as pd
import streamlit as st
from src.logger import logging
from src.exception import CustomException
from src.utils.utils import load_object


class PredictionPipeline:
    def __init__(self):
        self.preprocessor_obj_file_path=os.path.join('artifacts\Model','preprocessor.pkl')
        self.model_obj_file_path = os.path.join('artifacts\Model','model.pkl')
    
    def preprocess_data(self,df:pd.DataFrame):
        logging.info("\t\t\t (prediction_pipeline:preprocess_data) Started Preprocess the raw data \n")
        cols = ["gill-spacing" , "stem-root" , "stem-surface" , "veil-type" , "veil-color" , "spore-print-color","has-ring"]
        df.drop(columns= cols, inplace= True)
        df.columns = df.columns.str.replace('-', '_')
        df.rename(columns = {"does_bruise_or_bleed" : "does-bruise-or-bleed"} , inplace = True)
        if "class" in df.columns:
            df.drop(columns = ["class"] , axis=1 , inplace = True)
        return df 


    def predict(self,features , preprocess_feature = False):
        try:
            if preprocess_feature:
                features = self.preprocess_data(features)
            preprocessor = load_object(self.preprocessor_obj_file_path)
            model = load_object(self.model_obj_file_path)
            preprocessed_data = preprocessor.transform(features)
            prediction = model.predict(preprocessed_data)
            return prediction

        except Exception as e:
            logging.error("\t\t\t (prediction_pipeline) :"+str(CustomException(e , sys))+ "\n")
            raise CustomException(e , sys)
        

class CustomData:
    def __init__(self,cap_diameter:float,
                 cap_shape:str,
                 cap_surface:str,
                 cap_color:str,
                 does_bruise_or_bleed:str, 
                 gill_attachment:str,
                 gill_color:str,
                 stem_height:float,
                 stem_width:float,
                 stem_color:str,
                 ring_type:str,
                 habitat:str,
                 season:str):
        
        self.cap_diameter=cap_diameter
        self.cap_shape=cap_shape
        self.cap_surface=cap_surface
        self.cap_color= cap_color
        self.does_bruise_or_bleed= does_bruise_or_bleed
        self.gill_attachment= gill_attachment
        self.gill_color = gill_color
        self.stem_height = stem_height
        self.stem_width = stem_width
        self.stem_color = stem_color
        self.ring_type = ring_type
        self.habitat = habitat
        self.season = season
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'cap_diameter':[self.cap_diameter],
                'cap_shape':[self.cap_shape],
                'cap_surface':[self.cap_surface],
                'cap_color':[self.cap_color],
                'does-bruise-or-bleed':[self.does_bruise_or_bleed],
                'gill_attachment':[self.gill_attachment],
                'gill_color':[self.gill_color],
                'stem_height':[self.stem_height],
                'stem_width':[self.stem_width],
                'stem_color':[self.stem_color],
                'ring_type':[self.ring_type],
                'habitat':[self.habitat],
                'season'  : [self.season]
                }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)