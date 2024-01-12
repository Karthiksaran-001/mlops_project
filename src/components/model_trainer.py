import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from src.config import CONFIG_FILE_PATH
from src.utils import read_yaml
from src.utils.utils import save_object,evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import  XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


@dataclass 
class ModelTrainerConfig:
    config_values =  read_yaml(CONFIG_FILE_PATH)
    trained_model_file_path:Path = Path(config_values.model_trainer.model_saved_path)
    models = dict(config_values.model_trainer.models) 
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models= {'LogisticRegression':LogisticRegression(class_weight='balanced', random_state=42 , solver='liblinear'),'RandomForest':RandomForestClassifier(bootstrap= False, max_depth= 20, max_features= "log2", min_samples_leaf= 3, min_samples_split= 6, n_estimators= 60),'XgBoost':XGBClassifier(objective='binary:logistic',eval_metric='logloss'),'DecisionTree':DecisionTreeClassifier(criterion='gini', splitter='best'),'KNN' :KNeighborsClassifier(n_neighbors=5),'NavieBayes' :BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None),'SVM' : SVC(C=1.0 , class_weight='balanced')}
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.error("\t\t\t (DataTraining) :"+str(CustomException(e , sys))+ "\n")
            raise CustomException(e , sys)


        
    