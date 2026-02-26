# src/dermCNN/callbacks.py
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

def get_callbacks(mode='binary'):
    early = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # Zapisze np. 'best_model_binary.keras' lub 'best_model_malignant_only.keras'
    checkpoint = ModelCheckpoint(
        f"results/best_model_{mode}.keras",
        monitor="val_loss",
        save_best_only=True
    )

    logger = CSVLogger(f"results/training_log_{mode}.csv")

    return [early, checkpoint, logger]