import os
import sys
from sklearn.metrics import accuracy_score , classification_report
from urllib.parse import urlparse
from src.logger.logging import logging
from src.exception.exception import customException
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.utils.utils import load_object

@dataclass
class ModelEvaluationConfig:
    pass

class ModelEvaluation:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customException(e,sys)
