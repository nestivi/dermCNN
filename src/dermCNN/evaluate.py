"""Evaluation module for the DermCNN project.

This module evaluates trained models on the test dataset. It generates
predictions, computes the confusion matrix, and creates a detailed
classification report (precision, recall, f1-score) to assess model performance.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

from .data import load_dataframe, make_generators
from .config import MODEL_OUTPUT_PATH_STAGE1, MODEL_OUTPUT_PATH_STAGE2

def evaluate_model(mode: str = 'binary') -> None:
    """Evaluates the model and generates performance reports.

    Loads the test data and the corresponding trained model. Generates
    predictions, plots a confusion matrix using Seaborn, and saves both
    the plot and a text-based classification report to the 'results' directory.

    Args:
        mode (str): The classification mode. Either 'binary' (benign vs. malignant)
            or 'malignant_only' (classification of malignant types). Defaults to 'binary'.

    Returns:
        None

    Raises:
        ValueError: If an unsupported mode is provided.
        FileNotFoundError: If the corresponding model file is not found.
    """
    print(f"\n--- STARTING EVALUATION: ({mode.upper()}) ---")

    # Validate mode
    if mode not in ['binary', 'malignant_only']:
        raise ValueError(f"Unsupported mode: {mode}. Choose 'binary' or 'malignant_only'.")

    # 1. Determine the correct model path based on the mode
    if mode == 'binary':
        model_path = MODEL_OUTPUT_PATH_STAGE1
    else:
        model_path = MODEL_OUTPUT_PATH_STAGE2

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Error: Model file not found at {model_path}. Please train the model first."
        )

    # 2. Load data using the exact same logic to ensure identical test splits
    df = load_dataframe(mode=mode)
    
    # Unpack generators, keeping only the test generator
    _, test_gen = make_generators(df, mode=mode) 

    # 3. Load the pre-trained model into memory
    print(f"Loading model from: {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # 4. Generate predictions for the entire test set
    print("Evaluating test images... This may take a moment.")
    predictions = model.predict(test_gen)

    # 5. Process the results
    y_true = test_gen.classes  # Ground truth labels from the generator
    class_labels = list(test_gen.class_indices.keys())  # Class names (e.g., ['benign', 'malignant'])

    if mode == 'binary':
        # Convert sigmoid probabilities to class indices (0 or 1) using a 50% threshold
        y_pred = (predictions > 0.5).astype(int).flatten()
    else:
        # Convert softmax outputs to class indices by selecting the highest probability
        y_pred = np.argmax(predictions, axis=1)

    # 6. Plot the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    # Use seaborn.heatmap for a clean, colorful visualization
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=class_labels, yticklabels=class_labels
    )
    plt.title(f'AI Confusion Matrix - {mode.upper()}')
    plt.ylabel('True Diagnosis')
    plt.xlabel('Model Prediction')
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    cm_path = os.path.join("results", f"confusion_matrix_{mode}.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved successfully to: {cm_path}")
    
    # Display the plot window
    plt.show()

    # 7. Generate a textual classification report (Precision, Recall, F1-Score)
    print("\n--- CLASSIFICATION REPORT ---")
    print("You can copy this directly into your thesis!")
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)
    
    # Save the textual report to a file for later use
    report_path = os.path.join("results", f"classification_report_{mode}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Text report saved successfully to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the DermCNN model.")
    parser.add_argument(
        '--mode', type=str, choices=['binary', 'malignant_only'], default='binary',
        help="Choose evaluation mode: 'binary' or 'malignant_only'."
    )
    args = parser.parse_args()
    
    evaluate_model(mode=args.mode)