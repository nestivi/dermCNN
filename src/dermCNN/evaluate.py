"""Evaluation module for the DermCNN project.

This module evaluates trained models on the test dataset. It generates
predictions, computes the confusion matrix, and creates a detailed
classification report (precision, recall, f1-score) to assess model performance.
"""

import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

from .data import load_dataframe, make_generators
from .config import settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def evaluate_model(mode: str = 'binary') -> None:
    """Evaluates the model and generates performance reports.

    Loads the test data and the corresponding trained model. Generates
    predictions, plots a confusion matrix using Seaborn, and saves both
    the plot and a text-based classification report to the 'results' directory.
    """
    logging.info(f"--- STARTING EVALUATION: ({mode.upper()}) ---")

    if mode not in ['binary', 'malignant_only']:
        raise ValueError(f"Unsupported mode: {mode}. Choose 'binary' or 'malignant_only'.")

    model_path = settings.model_output_path_stage1 if mode == 'binary' else settings.model_output_path_stage2

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Error: Model file not found at {model_path}. Please train the model first."
        )

    df = load_dataframe(mode=mode)
    _, test_gen = make_generators(df, mode=mode) 

    logging.info(f"Loading model from: {model_path}...")
    model = tf.keras.models.load_model(model_path)
    logging.info("Evaluating test images...")

    predictions = model.predict(test_gen)
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    if mode == 'binary':
        y_pred = (predictions > settings.optimal_threshold).astype(int).flatten()
    else:
        y_pred = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=class_labels, yticklabels=class_labels
    )
    plt.title(f'Confusion Matrix - {mode.upper()}')
    plt.ylabel('True Diagnosis')
    plt.xlabel('Model Prediction')
    
    os.makedirs("results", exist_ok=True)
    cm_path = os.path.join("results", f"confusion_matrix_{mode}.png")
    plt.savefig(cm_path)
    logging.info(f"Confusion matrix plot saved successfully to: {cm_path}")
    plt.show()

    logging.info("--- CLASSIFICATION REPORT ---")
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)
    
    report_path = os.path.join("results", f"classification_report_{mode}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logging.info(f"Text report saved successfully to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the DermCNN model.")
    parser.add_argument(
        '--mode', type=str, choices=['binary', 'malignant_only'], default='binary',
        help="Choose evaluation mode: 'binary' or 'malignant_only'."
    )
    args = parser.parse_args()
    
    evaluate_model(mode=args.mode)