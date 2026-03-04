"""Gradio web interface for the DermCNN project.

This module provides a user-friendly graphical interface for the 
cascade classification system. It allows users to upload dermoscopy 
images and receive a two-stage AI diagnosis (benign/malignant, 
followed by the specific malignant tumor type if applicable).
"""

import os
from typing import Dict, Optional

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# --- CONFIGURATION ---
MODEL_STAGE1_PATH = os.path.join("results", "model_stage1_binary.keras")
MODEL_STAGE2_PATH = os.path.join("results", "model_stage2_malignant.keras")
IMG_SIZE = 224

# Stage 2 class names (alphabetical order as trained by Keras)
STAGE2_CLASSES = [
    'AK (Actinic Keratosis)', 
    'BCC (Basal Cell Carcinoma)', 
    'MEL (Melanoma)', 
    'SCC (Squamous Cell Carcinoma)'
]

# Global variables to hold models for lazy loading
model_stage1: Optional[tf.keras.Model] = None
model_stage2: Optional[tf.keras.Model] = None

def load_models() -> None:
    """Loads both stage 1 and stage 2 Keras models into memory.
    
    Uses lazy loading to avoid memory overhead until the first 
    prediction is requested by the user.
    
    Raises:
        FileNotFoundError: If the pre-trained model files do not exist.
    """
    global model_stage1, model_stage2
    
    if model_stage1 is None or model_stage2 is None:
        print("Loading models into memory... This may take a moment.")
        
        if not os.path.exists(MODEL_STAGE1_PATH) or not os.path.exists(MODEL_STAGE2_PATH):
            raise FileNotFoundError(
                "Model files are missing. Please run the training pipeline first."
            )
            
        model_stage1 = tf.keras.models.load_model(MODEL_STAGE1_PATH)
        model_stage2 = tf.keras.models.load_model(MODEL_STAGE2_PATH)
        print("Models loaded successfully!")

def predict_pipeline(image: np.ndarray) -> Dict[str, float]:
    """Processes the uploaded image through the cascade classification system.

    Args:
        image (np.ndarray): The uploaded image array from the Gradio interface.

    Returns:
        Dict[str, float]: A dictionary containing the predicted labels as keys 
            and their corresponding probabilities as values.
    """
    if image is None:
        return {"Error: No image uploaded.": 1.0}
    
    # Ensure models are loaded before making predictions
    try:
        load_models()
    except Exception as e:
        return {f"Model Loading Error: {str(e)}": 1.0}

    # 1. Image preprocessing for EfficientNet
    # Resize the image and create a batch of size 1.
    # Note: No division by 255.0 is applied, as EfficientNet handles scaling internally.
    img_resized = Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # 2. STAGE 1: Is it malignant?
    # Sigmoid outputs a single value between 0.0 (benign) and 1.0 (malignant)
    pred_stage1 = model_stage1.predict(img_array)[0][0]
    prob_malignant = float(pred_stage1)
    prob_benign = 1.0 - prob_malignant

    results: Dict[str, float] = {}

    # Cascade decision logic (using a 50% threshold)
    if prob_malignant < 0.5:
        # If the lesion is benign, terminate the diagnosis here
        results["🟢 BENIGN Lesion"] = prob_benign
        results["Malignancy Risk"] = prob_malignant
        return results
    else:
        # 3. STAGE 2: What type of malignant tumor is it?
        results["🔴 MALIGNANT Lesion - Probability"] = prob_malignant
        
        # Pass the image to the second model
        pred_stage2 = model_stage2.predict(img_array)[0]
        
        # Map the Stage 2 results to their respective class names
        for i, class_name in enumerate(STAGE2_CLASSES):
            # Using indentation for better UI readability in the Gradio label
            results[f"   ↳ {class_name}"] = float(pred_stage2[i])
            
        return results

# --- USER INTERFACE ---
# Construct the Gradio web view
interface = gr.Interface(
    fn=predict_pipeline,
    inputs=gr.Image(label="Upload dermoscopy image of a skin lesion"),
    outputs=gr.Label(num_top_classes=5, label="AI Diagnosis Result"),
    title="DermCNN AI - Cascade Classification System",
    description="This algorithm first evaluates whether the skin lesion is malignant. If it is, the system triggers the second module to identify the specific type of skin cancer."
)

if __name__ == "__main__":
    # Launch the local web server
    interface.launch()