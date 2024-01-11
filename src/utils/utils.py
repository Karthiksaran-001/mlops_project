import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report  , accuracy_score 

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
            cv_accuracy = cross_val_score(model , X_test , y_test , cv = 5)
            
            logging.info("\t\t\t Cross Validataion Scores are : {} average CV score is :{}\n".format(cv_accuracy , cv_accuracy.mean()))

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

    