# data.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import BASE_DIR, CSV_PATH, IMG_SIZE, BATCH_SIZE


def load_dataframe():
    df = pd.read_csv(CSV_PATH)

    df = df[(df["MEL"] == 1) | (df["NV"] == 1)]
    df["label"] = df["MEL"].apply(lambda x: "MEL" if x == 1 else "NV")
    df["filepath"] = df["image"].apply(lambda x: os.path.join(BASE_DIR, x + ".jpg"))

    df = df[df["filepath"].apply(os.path.exists)]
    return df


def make_generators(df):
    train_df, test_df = train_test_split(df, test_size=0.2,
                                         stratify=df["label"], random_state=42)

    train_datagen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1/255.0)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="filepath", y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="binary", batch_size=BATCH_SIZE, shuffle=True
    )

    test_gen = test_datagen.flow_from_dataframe(
        test_df, x_col="filepath", y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="binary", batch_size=BATCH_SIZE
    )

    return train_gen, test_gen
