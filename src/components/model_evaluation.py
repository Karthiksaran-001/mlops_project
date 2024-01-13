import mlflow
import os
import sys
import mlflow.sklearn 
from pathlib import Path
from src.logger import logging
from src.exception import CustomException
from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report  , confusion_matrix  , accuracy_score
from src.config import CONFIG_FILE_PATH
from src.utils import read_yaml
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluation:
    def __init__(self):
        logging.info("evaluation started")
        self.config_values =  read_yaml(CONFIG_FILE_PATH)
        self.model_path: Path = self.config_values.model_trainer.model_saved_path 

    def plot_confusion_matrix(self,conf_matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=.5)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")


    def eval_metrics(self,actual,pred):
        acc_score = accuracy_score(actual , pred)
        cm = confusion_matrix(actual , pred)
        class_report = classification_report(actual , pred)
        return acc_score, cm , class_report

    def initiate_model_evaluation(self,train_array,test_array):
        try:
             X_test,y_test=(test_array[:,:-1], test_array[:,-1])
             model=load_object(self.model_path)

             #mlflow.set_registry_uri("")
             
             logging.info("\t\t\t model has register\n")

             tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

             print(tracking_url_type_store)



             with mlflow.start_run():

                prediction=model.predict(X_test)
                cv_accuracy = cross_val_score(model , X_test , y_test , cv = 5)
                cv_accuracy = cv_accuracy.mean()

                (accuracy_score,cm,class_report)=self.eval_metrics(y_test,prediction)

                mlflow.log_metric("accuracy score", accuracy_score)
                mlflow.log_metric("cross validation average score", cv_accuracy)
                mlflow.log_text(class_report, "classification_report.txt")
                self.plot_confusion_matrix(cm)
                mlflow.log_artifact("confusion_matrix.png")
                if os.path.exists("confusion_matrix.png"):
                    os.remove("confusion_matrix.png")

                 # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")
                logging.info("\t\t\t Model Evaluation finished\n")


        except Exception as e:
            logging.error("\t\t\t (Model-Evaluation) :"+str(CustomException(e , sys))+ "\n")
            raise CustomException(e,sys)
