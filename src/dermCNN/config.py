"""Configuration file for the DermCNN project.

This module contains all the global constants, hyperparameter settings,
class definitions, and file paths used across the machine learning pipeline.
"""

import os

# --- Directory Paths ---
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
BASE_DIR = os.path.join(BASE_PROJECT_DIR, "ISIC_2019_Training_Input")
CSV_PATH = os.path.join(BASE_PROJECT_DIR, "ISIC_2019_Training_GroundTruth.csv")

# --- Model Hyperparameters ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# --- Class Definitions ---
# List of all original classes in the ISIC 2019 dataset
CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

# Stage 1: Binary classification split (Benign vs. Malignant)
BENIGN_CLASSES = ['NV', 'BKL', 'DF', 'VASC']
MALIGNANT_CLASSES = ['MEL', 'BCC', 'AK', 'SCC']

# --- Output Paths ---
# Save paths for the trained Keras models
MODEL_OUTPUT_PATH_STAGE1 = os.path.join("results", "model_stage1_binary.keras")
MODEL_OUTPUT_PATH_STAGE2 = os.path.join("results", "model_stage2_malignant.keras")