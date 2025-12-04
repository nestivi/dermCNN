import os

# config.py

# Path to the base project directory
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Paths to data
BASE_DIR = os.path.join(BASE_PROJECT_DIR, "data", "ISIC_2019_Training_Input")
CSV_PATH = os.path.join(BASE_PROJECT_DIR, "data", "ISIC_2019_Training_GroundTruth.csv")

# Model parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Model output path
RESULTS_PATH = os.path.join(BASE_PROJECT_DIR, "results")
os.makedirs(RESULTS_PATH, exist_ok=True)
MODEL_OUTPUT_PATH = os.path.join(BASE_PROJECT_DIR, "results", "model_isic_cnn.h5")