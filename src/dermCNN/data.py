"""Data processing module for the DermCNN project.

This module handles loading the dataset from a CSV file, filtering valid images,
assigning appropriate labels based on the classification stage, and creating
Keras ImageDataGenerators for training and testing with data augmentation.
"""

import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DataFrameIterator

from .config import (
    BASE_DIR, 
    CSV_PATH, 
    IMG_SIZE, 
    BATCH_SIZE, 
    CLASSES, 
    BENIGN_CLASSES, 
    MALIGNANT_CLASSES
)

def load_dataframe(mode: str = 'binary') -> pd.DataFrame:
    """Loads and processes the dataset based on the specified classification mode.

    Args:
        mode (str): The classification mode. Either 'binary' (benign vs. malignant)
            or 'malignant_only' (classification of malignant types). Defaults to 'binary'.

    Returns:
        pd.DataFrame: A processed DataFrame containing file paths and assigned labels.

    Raises:
        FileNotFoundError: If the ground truth CSV file is not found at CSV_PATH.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    
    # 1. Decode one-hot encoded classes to a single original label
    df["original_label"] = df[CLASSES].idxmax(axis=1)
    
    # Construct full file paths and filter out missing images
    df["filepath"] = df["image"].apply(lambda x: os.path.join(BASE_DIR, x + ".jpg"))
    df = df[df["filepath"].apply(os.path.exists)]

    # 2. Process labels based on the selected classification stage
    if mode == 'binary':
        # Group into 'benign' and 'malignant'
        df['label'] = df['original_label'].apply(
            lambda x: 'benign' if x in BENIGN_CLASSES else 'malignant'
        )
        print("Stage 1 Mode: Binary Classification (Benign vs. Malignant).")
        
    elif mode == 'malignant_only':
        # Filter the DataFrame to keep only malignant cases
        df = df[df['original_label'].isin(MALIGNANT_CLASSES)].copy()
        df['label'] = df['original_label']
        print("Stage 2 Mode: Multi-class Malignant Classification.")

    print(f"Data distribution:\n{df['label'].value_counts()}")
    return df

def make_generators(df: pd.DataFrame, mode: str = 'binary') -> Tuple[DataFrameIterator, DataFrameIterator]:
    """Creates Keras image data generators for training and validation.

    Splits the provided DataFrame into training and testing sets while maintaining
    class distribution. Applies data augmentation to the training set.

    Args:
        df (pd.DataFrame): The DataFrame containing image paths and labels.
        mode (str): The classification mode. Determines the class_mode for the generator.
            Defaults to 'binary'.

    Returns:
        Tuple[DataFrameIterator, DataFrameIterator]: A tuple containing 
        (train_generator, test_generator).
    """
    # Stratify ensures both sets have the exact same proportion of classes
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    # Note: No 'rescale' is applied here because EfficientNet models expect 
    # input values in the range [0, 255] and handle scaling internally.
    train_datagen = ImageDataGenerator(
        rotation_range=40, 
        width_shift_range=0.2, 
        height_shift_range=0.2,
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True, 
        vertical_flip=True, 
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator()

    # Set the appropriate class_mode based on the stage
    class_mode = "binary" if mode == 'binary' else "categorical"

    train_gen = train_datagen.flow_from_dataframe(
        train_df, 
        x_col="filepath", 
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode=class_mode, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )

    test_gen = test_datagen.flow_from_dataframe(
        test_df, 
        x_col="filepath", 
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode=class_mode, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    return train_gen, test_gen