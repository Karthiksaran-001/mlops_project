import os
import sys
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customException
from dataclasses import dataclass
from pathlib import Path
from src.utils.utils import save_object , evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

@dataclass
class ModelTrainerConfig:
    pass


class ModelTrainer:
    def __init__(self):
        pass
    def initate_model_trainer(self):
        try:
            pass
        except Exception as e:
            logging.error()
            raise customException(e , sys)
