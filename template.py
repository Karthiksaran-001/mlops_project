import os
import sys
from datetime import datetime
from pathlib import Path 


list_of_files = [
    ".github/workflows/.gitkeep",
    "config/config.yaml",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/config/__init__.py",
    "src/config/configuration.py",
    "src/utils/__init__.py",
    "src/utils/utils.py",
    "src/logger/__init__.py",
    "src/exception/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "_init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiments/experiment.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    fildir , filename = os.path.split(filepath)
    if fildir != "":
        os.makedirs(fildir , exist_ok = True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) ==0):
        with open(filepath , "w"):
            pass ## create an empty file 
        

