import os

# config.py

# Path to the base project directory
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

# Paths to data
BASE_DIR = os.path.join(BASE_PROJECT_DIR, "ISIC_2019_Training_Input", "ISIC_2019_Training_Input")
CSV_PATH = os.path.join(BASE_PROJECT_DIR, "ISIC_2019_Training_GroundTruth.csv")

# Model parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Model output path
MODEL_OUTPUT_PATH = os.path.join(BASE_PROJECT_DIR, "results", "model_isic_cnn.h5")