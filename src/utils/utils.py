import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report  , confusion_matrix

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.error("\t\t\t (Utils) :"+str(CustomException(e , sys))+ "\n")
        raise CustomException(e , sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            model_classification_report = classification_report(y_test,y_test_pred)
            logging.info("\t\t\t Classification report for {} is :  \n%s".format(list(models.keys())[i]),model_classification_report)
            cm = confusion_matrix(y_test , y_test_pred)
            logging.info("\t\t\tConfusion Matrix for {} is : \n%s".format(list(models.keys())[i]),cm)
            cv_accuracy = cross_val_score(model , X_test , y_test , cv = 5)
            
            logging.info("\t\t\t Cross Validataion Scores are : {} average CV score is :{:.2%}\n".format(cv_accuracy , cv_accuracy.mean()))

            test_model_score = cv_accuracy.mean()

            report[list(models.keys())[i]] =  test_model_score
            logging.info("*"*200)

        return report

    except Exception as e:
        logging.error("\t\t\t (Utils) :"+str(CustomException(e , sys))+ "\n")
        raise CustomException(e , sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error("\t\t\t (Utils) :"+str(CustomException(e , sys))+ "\n")
        raise CustomException(e , sys)


class CustomOHETransformer:
    def __init__(self, oh_encoder_path):
        self.oh_encoder_path = oh_encoder_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        encoder = load_object(self.oh_encoder_path)
        ohe = encoder.transform(X[encoder.feature_names_in_])
        new_df = pd.DataFrame(ohe, columns=encoder.get_feature_names_out())
        X.reset_index(drop=True, inplace=True)
        new_df.reset_index(drop=True, inplace=True)
        X.drop(columns=encoder.feature_names_in, axis=1, inplace=True)
        X = pd.concat([X, new_df], axis=1)
        return X
    
    def fit_transform(self, X):
        encoder = load_object(self.oh_encoder_path)
        ohe = encoder.transform(X[encoder.feature_names_in_])
        new_df = pd.DataFrame(ohe, columns=encoder.get_feature_names_out())
        X.reset_index(drop=True, inplace=True)
        new_df.reset_index(drop=True, inplace=True)
        X.drop(columns=encoder.feature_names_in, axis=1, inplace=True)
        X = pd.concat([X, new_df], axis=1)
        return X


    