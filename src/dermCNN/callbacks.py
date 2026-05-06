"""Keras callbacks module for the DermCNN project.

This module provides a configured list of callbacks to monitor and control
the training process, including early stopping, model checkpointing,
and CSV logging.
"""

import os
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint
from .config import settings

def get_callbacks(mode: str = 'binary') -> tuple[Callback, ...]:
    """Creates and configures Keras callbacks for model training."""
    early = EarlyStopping(
        monitor="val_loss",
        patience=settings.early_stopping_patience,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join("results", f"best_model_{mode}.keras"),
        monitor="val_loss",
        save_best_only=True
    )

    # Stream epoch results to a CSV file for later analysis and plotting
    logger = CSVLogger(
        filename=os.path.join("results", f"training_log_{mode}.csv")
    )

    return (early, checkpoint, logger)