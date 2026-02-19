# app_gradio.py

import os
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from src.dermCNN.config import MODEL_OUTPUT_PATH, IMG_SIZE, CLASSES

print("Loading model...")
model = tf.keras.models.load_model(MODEL_OUTPUT_PATH)
print("Model loaded.")

def predict_skin_lesion(image):
    # Gradio send image as numpy array, convert to PIL Image for resizing
    image = Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, IMG_SIZE, IMG_SIZE, 3)

    # Prediction
    prediction = model.predict(img_array)[0]
    
    # Create a dictionary of class probabilities 
    return {CLASSES[i]: float(prediction[i]) for i in range(len(CLASSES))}

# UI
interface = gr.Interface(
    fn=predict_skin_lesion,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="DermCNN - Classification of Skin Lesions in Dermatoscopic Images Using Artificial Intelligence",
    description="Upload a dermoscopic image to identify the type of skin lesion.",
    examples=["tests/ISIC_2019_Training_Input/ISIC_0000000.jpg"] if os.path.exists("tests/ISIC_2019_Training_Input/ISIC_0000000.jpg") else None
)

if __name__ == "__main__":
    interface.launch()