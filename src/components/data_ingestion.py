import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import traceback
from pathlib import Path


@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts\Data","Data.csv")
    train_data_path:str=os.path.join("artifacts\Data","train.csv")
    test_data_path:str=os.path.join("artifacts\Data","test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initate_data_ingestion(self):
        logging.info("\t\t\t Step 1 : Data Ingestion Started \n")
        try:
            os.makedirs("artifacts\Data" , exist_ok=True)
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("\t\t\t Total Shape of Data : {}rows {}columns \n".format(df.shape[0], df.shape[1]))
            logging.info("\t\t\t Total Duplicate records : {} \n".format(df.duplicated().sum()))
            cols = ["gill-spacing" , "stem-root" , "stem-surface" , "veil-type" , "veil-color" , "spore-print-color","has-ring"]
            df.drop_duplicates(inplace=True)
            df.drop(columns= cols, inplace= True)
            logging.info("\t\t\t After delete & remove the duplicate and unwanted columns the overall shape is {} \n".format(df.shape))
            df["class"] = df["class"].replace({"p" : 0 , "e" : 1})
            df.columns = df.columns.str.replace('-', '_')
            df.rename(columns = {"does_bruise_or_bleed" : "does-bruise-or-bleed"} , inplace = True)
            train_data , test_data = train_test_split(df , test_size=0.25)
            logging.info("\t\t\t Train Data Shape is : {} Train Data Shape is : {} \n".format(train_data.shape , test_data.shape))
            train_data.to_csv(self.ingestion_config.train_data_path , index = False)
            test_data.to_csv(self.ingestion_config.test_data_path , index = False)
            logging.info("\t\t\t Data Ingestion Completed \n")
            return (self.ingestion_config.train_data_path , self.ingestion_config.test_data_path)

        except Exception as e:
            logging.error("\t\t\t (Utils) :"+str(CustomException(e , sys))+ "\n")
            raise CustomException(e , sys)

