"""Keras callbacks module for the DermCNN project.

This module provides a configured list of callbacks to monitor and control
the training process, including early stopping, model checkpointing,
and CSV logging.
"""

from typing import List
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint

def get_callbacks(mode: str = 'binary') -> List[Callback]:
    """Creates and configures Keras callbacks for model training.

    Args:
        mode (str): The classification mode. Determines the output filenames
            for the saved models and logs. Defaults to 'binary'.

    Returns:
        List[Callback]: A list of configured Keras callback instances.
    """
    # Stop training when validation loss stops improving for 5 epochs.
    # Restores model weights from the epoch with the best validation loss.
    early = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # Save the best model dynamically based on the current mode
    # e.g., 'results/best_model_binary.keras' or 'results/best_model_malignant_only.keras'
    checkpoint = ModelCheckpoint(
        filepath=f"results/best_model_{mode}.keras",
        monitor="val_loss",
        save_best_only=True
    )

    # Stream epoch results to a CSV file for later analysis and plotting
    logger = CSVLogger(filename=f"results/training_log_{mode}.csv")

    return [early, checkpoint, logger]