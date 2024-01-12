from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    train_data_file: Path
    test_data_file: Path
    remove_cols: List[str]
    output_mapping: dict[str , int]


@dataclass(frozen=True)
class DataTransformationConfig:
    ohe_model_path: Path
    updated_base_model_path: Path
    target_col: str
    numerical_cols:  List[str]
    categorical_cols: List[str]
    pass_through: List[str]
    cap_shape:  List[str]
    cap_surface:  List[str]
    cap_color:  List[str]
    gill_attachment:  List[str]
    gill_color:  List[str]
    stem_color:  List[str]
    ring_type:   List[str]
    habitat:  List[str]

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    model_saved_path: Path
    models: Dict[str,float]