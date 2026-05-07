"""Data processing module for the DermCNN project.

This module handles loading the dataset from a CSV file, filtering valid images,
assigning appropriate labels based on the classification stage, and creating
Keras ImageDataGenerators for training and testing with data augmentation.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DataFrameIterator

from .config import (
    settings, 
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
    """
    if not os.path.exists(settings.csv_path):
        raise FileNotFoundError(f"CSV file not found: {settings.csv_path}")
    
    df = pd.read_csv(settings.csv_path)
    df["original_label"] = df[list(CLASSES)].idxmax(axis=1)
    
    df["filepath"] = df["image"].apply(lambda x: os.path.join(settings.base_dir, x + ".jpg"))
    df = df[df["filepath"].apply(os.path.exists)]

    if settings.debug_mode:
        print("DEBUG MODE: Train only on 100 random samples!")
        df = df.sample(n = 100, random_state=settings.random_seed).reset_index(drop=True)

    if mode == 'binary':
        df['label'] = df['original_label'].apply(
            lambda x: 'benign' if x in BENIGN_CLASSES else 'malignant'
        )
        print("Stage 1 Mode: Binary Classification (Benign vs. Malignant).")
        
    elif mode == 'malignant_only':
        df = df[df['original_label'].isin(MALIGNANT_CLASSES)].copy()
        df['label'] = df['original_label']
        print("Stage 2 Mode: Multi-class Malignant Classification.")

    print(f"Data distribution:\n{df['label'].value_counts()}")
    return df

def make_generators(df: pd.DataFrame, mode: str = 'binary') -> tuple[DataFrameIterator, DataFrameIterator]:
    """Creates Keras image data generators for training and validation.

    Args:
        df (pd.DataFrame): The DataFrame containing image paths and labels.
        mode (str): The classification mode. Determines the class_mode for the generator.
            Defaults to 'binary'.
    """
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=settings.random_seed
    )

    if settings.use_data_augumentation:
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
    else:
        train_datagen = ImageDataGenerator()
    
    test_datagen = ImageDataGenerator()
    class_mode = "binary" if mode == 'binary' else "categorical"

    train_gen = train_datagen.flow_from_dataframe(
        train_df, 
        x_col="filepath", 
        y_col="label",
        target_size=(settings.img_size, settings.img_size),
        class_mode=class_mode, 
        batch_size=settings.batch_size, 
        shuffle=True
    )

    test_gen = test_datagen.flow_from_dataframe(
        test_df, 
        x_col="filepath", 
        y_col="label",
        target_size=(settings.img_size, settings.img_size),
        class_mode=class_mode, 
        batch_size=settings.batch_size, 
        shuffle=False
    )

    return train_gen, test_gen