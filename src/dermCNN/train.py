"""Main training module for the DermCNN project.

This script orchestrates the entire training pipeline: loading data,
building the model architecture, setting up callbacks, executing the 
training loop, saving the final model, and plotting the training history.
"""

import os
from tensorflow.keras.callbacks import History

from .data import load_dataframe, make_generators
from .model import build_model
from .callbacks import get_callbacks
from .plot import plot_history
from .config import EPOCHS, MODEL_OUTPUT_PATH_STAGE1, MODEL_OUTPUT_PATH_STAGE2

def train(mode: str = 'binary') -> History:
    """Executes the training pipeline for the specified mode.

    Args:
        mode (str): The classification mode. Either 'binary' (benign vs. malignant)
            or 'malignant_only' (classification of malignant types). Defaults to 'binary'.

    Returns:
        History: The Keras History object containing training metrics across epochs.

    Raises:
        ValueError: If an unsupported mode is provided, or if the loaded DataFrame is empty.
    """
    print(f"\n--- STARTING TRAINING: ({mode.upper()}) ---")
    
    # Validate mode
    if mode not in ['binary', 'malignant_only']:
        raise ValueError(f"Unsupported mode: {mode}. Choose 'binary' or 'malignant_only'.")

    # Determine the correct output path based on the classification mode
    if mode == 'binary':
        output_path = MODEL_OUTPUT_PATH_STAGE1
    else:
        output_path = MODEL_OUTPUT_PATH_STAGE2
    
    # Load and validate the dataset
    df = load_dataframe(mode=mode)

    if df.empty:
        raise ValueError("DataFrame is empty. Please check the CSV_PATH in config.py.")
    
    print(f"Successfully loaded {len(df)} samples from the dataset.")
    
    # Initialize data generators and build the CNN architecture
    train_gen, test_gen = make_generators(df, mode=mode)
    model = build_model(mode=mode)
    
    # Retrieve configured callbacks to prevent overfitting and save progress
    callbacks = get_callbacks(mode=mode)

    # Execute the training loop
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Ensure the output directory exists and save the trained model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved successfully at: {output_path}")
    
    # Generate and save the training history plots
    plot_history(history, mode=mode)

    return history