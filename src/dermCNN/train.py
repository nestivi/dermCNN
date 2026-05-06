"""Main training module for the DermCNN project.

This script orchestrates the entire training pipeline: loading data,
building the model architecture, setting up callbacks, executing the 
training loop, saving the final model, and plotting the training history.
"""

import os
import logging
from tensorflow.keras.callbacks import History

from .data import load_dataframe, make_generators
from .model import build_model
from .callbacks import get_callbacks
from .plot import plot_history
from .config import settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train(mode: str = 'binary') -> History:
    """Executes the training pipeline for the specified mode."""
    logging.info(f"--- STARTING TRAINING: ({mode.upper()}) ---")
    
    if mode not in ['binary', 'malignant_only']:
        raise ValueError(f"Unsupported mode: {mode}. Choose 'binary' or 'malignant_only'.")
    
    output_path = settings.model_output_path_stage1 if mode == 'binary' else settings.model_output_path_stage2
    
    df = load_dataframe(mode=mode)
    if df.empty:
        raise ValueError("DataFrame is empty. Please check the CSV_PATH in your config/.env")
    
    logging.info(f"Successfully loaded {len(df)} samples from the dataset.")
    
    train_gen, test_gen = make_generators(df, mode=mode)
    model = build_model(mode=mode)
    callbacks = get_callbacks(mode=mode)

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=settings.epochs,
        callbacks=callbacks
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    logging.info(f"Model saved successfully at: {output_path}")
    plot_history(history, mode=mode)

    return history