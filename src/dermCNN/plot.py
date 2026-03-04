"""Visualization module for the DermCNN project.

This module provides utility functions to generate, save, and display
training history plots (accuracy and loss curves) over epochs.
"""

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History

def plot_history(history: History, mode: str = 'binary') -> None:
    """Plots and saves the training and validation metrics.

    Extracts 'accuracy', 'val_accuracy', 'loss', and 'val_loss' from the 
    Keras History object. Generates a figure with two subplots and saves 
    it dynamically based on the current training mode.

    Args:
        history (History): The Keras History object returned by model.fit().
        mode (str): The classification mode. Determines the title and filename 
            of the saved plot. Defaults to 'binary'.

    Returns:
        None
    """
    # Extract metrics from the history dictionary
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # Define the x-axis representing the number of epochs
    epochs = range(1, len(acc) + 1)

    # Initialize the figure size
    plt.figure(figsize=(12, 5))

    # --- Subplot 1: Accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.title(f"Accuracy ({mode})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # --- Subplot 2: Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title(f"Loss ({mode})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Save the plot dynamically to prevent overwriting results from other stages
    output_filepath = f"results/training_plot_{mode}.png"
    plt.savefig(output_filepath)
    print(f"Training plot saved successfully: {output_filepath}")
    
    # Display the plot in the graphical interface
    plt.show()