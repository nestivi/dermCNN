"""Gradio web interface for the DermCNN project.

This module provides a user-friendly graphical interface for the 
cascade classification system. It allows users to upload dermoscopy 
images and receive a two-stage AI diagnosis (benign/malignant, 
followed by the specific malignant tumor type if applicable).
"""

import os
from enum import Enum
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from dermCNN.config import settings

class DiagnosisState(str, Enum):
    BENIGN = "BENIGN Lesion"
    MALIGNANT = "MALIGNANT Lesion - Probability"
    RISK = "Malignancy Risk"
    ERROR = "Error"

# Stage 2 class names
STAGE2_CLASSES = (
    'AK (Actinic Keratosis)', 
    'BCC (Basal Cell Carcinoma)', 
    'MEL (Melanoma)', 
    'SCC (Squamous Cell Carcinoma)'
)

model_stage1: tf.keras.Model | None = None
model_stage2: tf.keras.Model | None= None

def load_models() -> None:
    """Loads both stage 1 and stage 2 Keras models into memory."""
    global model_stage1, model_stage2
    
    if model_stage1 is None or model_stage2 is None:
        print("Loading models into memory... This may take a moment.")
        
        if not os.path.exists(settings.model_output_path_stage1) or not os.path.exists(settings.model_output_path_stage2):
            raise FileNotFoundError(
                "Model files are missing. Please run the training pipeline first."
            )
            
            
        model_stage1 = tf.keras.models.load_model(settings.model_output_path_stage1)
        model_stage2 = tf.keras.models.load_model(settings.model_output_path_stage2)
        print("Models loaded successfully!")

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Resizes and formats the image array for model input."""
    img_resized = Image.fromarray(image).resize((settings.img_size, settings.img_size))
    img_array = np.array(img_resized)
    return np.expand_dims(img_array, axis=0)

def predict_pipeline(image: np.ndarray) -> dict[str, float]:
    """Processes the uploaded image through the cascade classification system.

    Returns:
        dict[str, float]: A dictionary containing the predicted labels as keys 
            and their corresponding probabilities as values.
    """
    if image is None:
        return {DiagnosisState.ERROR: 1.0}, {}
    
    try:
        load_models()
    except Exception as e:
        return {f"{DiagnosisState.ERROR}: Model loading - {str(e)}": 1.0}, {}

    img_array = preprocess_image(image)

    pred_stage1 = model_stage1.predict(img_array)[0][0]
    prob_malignant = float(pred_stage1)
    prob_benign = 1.0 - prob_malignant

    stage1_results = {
        DiagnosisState.BENIGN.value: prob_benign,
        DiagnosisState.MALIGNANT.value: prob_malignant
    }

    stage2_results: dict[str, float] = {}

    if prob_malignant >= 0.5:
        pred_stage2 = model_stage2.predict(img_array)[0]
        for i, class_name in enumerate(STAGE2_CLASSES):
            stage2_results[class_name] = float(pred_stage2[i])
    
    return stage1_results, stage2_results

# --- USER INTERFACE ---
interface = gr.Interface(
    fn=predict_pipeline,
    inputs=gr.Image(label="Upload dermoscopy image of a skin lesion"),
    outputs=[
        gr.Label(num_top_classes=2, label="AI Binary Diagnosis Result"),
        gr.Label(num_top_classes=4, label="Malignant Tumor Type")
    ],
    title="DermCNN AI - Cascade Classification System",
    description="This algorithm first evaluates whether the skin lesion is malignant. If it is, the system triggers the second module to identify the specific type of skin cancer."
)

if __name__ == "__main__":
    interface.launch()