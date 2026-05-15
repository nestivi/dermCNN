"""Keras callbacks module for the DermCNN project.

This module provides a configured list of callbacks to monitor and control
the training process, including early stopping, model checkpointing,
and CSV logging.
"""

import os
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from .config import settings

def get_callbacks(mode: str = 'binary') -> tuple[Callback, ...]:
    """Creates and configures Keras callbacks for model training."""

    if mode == 'binary':
        monitor_metric = "val_loss"
        callback_mode = "min"
    else:
        monitor_metric = "val_macro_f1_score"
        callback_mode = "max"

    early = EarlyStopping(
        monitor=monitor_metric,
        patience=settings.early_stopping_patience,
        restore_best_weights=True,
        mode=callback_mode
    )

    reduce_lr = ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=settings.lr_reduce_factor,
        patience=settings.lr_reduce_patience,
        min_lr=1e-6,
        mode=callback_mode,
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join("results", f"best_model_{mode}.keras"),
        monitor=monitor_metric,
        save_best_only=True,
        mode=callback_mode
    )

    # Stream epoch results to a CSV file for later analysis and plotting
    logger = CSVLogger(
        filename=os.path.join("results", f"training_log_{mode}.csv")
    )

    return (early, reduce_lr, checkpoint, logger)