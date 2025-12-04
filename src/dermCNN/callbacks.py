# callbacks.py

from .config import RESULTS_PATH
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

def get_callbacks():
    early = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        "results/best_model.h5",
        monitor="val_loss",
        save_best_only=True
    )

    logger = CSVLogger(f'{RESULTS_PATH}/training_log.csv')

    return [early, checkpoint, logger]
