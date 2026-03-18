"""Model architecture module for the DermCNN project.

This module defines the Convolutional Neural Network (CNN) architecture
using Transfer Learning. It utilizes a pre-trained EfficientNetB0 base
with a custom classification head tailored for either binary or multi-class
classification stages.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications

from .config import IMG_SIZE

def build_model(mode: str = 'binary') -> models.Sequential:
    """Builds and compiles the Keras Sequential model.

    Constructs a CNN using EfficientNetB0 (pre-trained on ImageNet) as a 
    feature extractor. The base model is frozen. A custom classification head 
    is appended based on the specified classification mode.

    Args:
        mode (str): The classification mode. Valid options are 'binary' 
            (benign vs. malignant) or 'malignant_only' (classification of 
            malignant types). Defaults to 'binary'.

    Returns:
        models.Sequential: A compiled Keras model ready for training.
        
    Raises:
        ValueError: If an unsupported mode is provided.
    """
    if mode not in ['binary', 'malignant_only']:
        raise ValueError(f"Unsupported mode: {mode}. Choose 'binary' or 'malignant_only'.")

    # Load pre-trained EfficientNetB0 without the top classification layer
    base_model = applications.EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    base_model.trainable = False  # Freeze the base model to retain pre-trained ImageNet features

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3)
    ])

    if mode == 'binary':
        model.add(layers.Dense(1, activation='sigmoid'))
        loss_fn = "binary_crossentropy"
    else:
        model.add(layers.Dense(4, activation='softmax'))
        loss_fn = "categorical_crossentropy"
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer, 
        loss=loss_fn, 
        metrics=["accuracy"]
    )
    
    return model