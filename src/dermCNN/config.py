"""Configuration file for the DermCNN project.

This module contains all the global constants, hyperparameter settings,
class definitions, and file paths used across the machine learning pipeline.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

class Settings(BaseSettings):
    # --- Directory Paths ---
    base_project_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
    base_dir: str = os.path.join(base_project_dir, "ISIC_2019_Training_Input")
    csv_path: str = os.path.join(base_project_dir, "ISIC_2019_Training_GroundTruth.csv")

    # --- Output Paths ---
    model_output_path_stage1: str = os.path.join("results", "model_stage1_binary.keras")
    model_output_path_stage2: str = os.path.join("results", "model_stage2_malignant.keras")

    # --- Model Hyperparameters ---
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 40
    early_stopping_patience: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

# --- Class Definitions ---
# Tuple of all original classes in the ISIC 2019 dataset
CLASSES = ('MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC')
BENIGN_CLASSES = ('NV', 'BKL', 'DF', 'VASC')
MALIGNANT_CLASSES = ('MEL', 'BCC', 'AK', 'SCC')
