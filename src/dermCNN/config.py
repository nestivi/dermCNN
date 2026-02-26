# onfig.py
import os

BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
BASE_DIR = os.path.join(BASE_PROJECT_DIR, "ISIC_2019_Training_Input")
CSV_PATH = os.path.join(BASE_PROJECT_DIR, "ISIC_2019_Training_GroundTruth.csv")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Lista wszystkich oryginalnych klas
CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

# NOWE: Podział na łagodne i złośliwe
BENIGN_CLASSES = ['NV', 'BKL', 'DF', 'VASC']
MALIGNANT_CLASSES = ['MEL', 'BCC', 'AK', 'SCC']

# Ścieżka zapisu dla pierwszego modelu (Etap 1)
MODEL_OUTPUT_PATH_STAGE1 = os.path.join("results", "model_stage1_binary.keras")
MODEL_OUTPUT_PATH_STAGE2 = os.path.join("results", "model_stage2_malignant.keras")
