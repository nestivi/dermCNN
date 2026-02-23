# src/dermCNN/data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import BASE_DIR, CSV_PATH, IMG_SIZE, BATCH_SIZE, CLASSES, BENIGN_CLASSES, MALIGNANT_CLASSES

def load_dataframe(mode='binary'):
    """
    Ładuje dane i przypisuje etykiety w zależności od wybranego trybu.
    mode: 'binary' (łagodne vs złośliwe) lub 'malignant_only' (tylko 4 klasy złośliwe)
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    
    # 1. Odczytanie oryginalnej klasy
    df["original_label"] = df[CLASSES].idxmax(axis=1)
    df["filepath"] = df["image"].apply(lambda x: os.path.join(BASE_DIR, x + ".jpg"))
    df = df[df["filepath"].apply(os.path.exists)]

    # 2. Przetwarzanie na podstawie wybranego Etapu
    if mode == 'binary':
        # Grupujemy: jeśli klasa jest na liście BENIGN, oznacz jako 'benign', w przeciwnym razie 'malignant'
        df['label'] = df['original_label'].apply(
            lambda x: 'benign' if x in BENIGN_CLASSES else 'malignant'
        )
        print("Tryb Etap 1: Klasyfikacja binarna (Łagodne vs Złośliwe).")
        
    elif mode == 'malignant_only':
        # Filtrujemy DataFrame, aby zostawić tylko zmiany złośliwe
        df = df[df['original_label'].isin(MALIGNANT_CLASSES)].copy()
        df['label'] = df['original_label']
        print("Tryb Etap 2: Klasyfikacja rodzajów zmian złośliwych.")

    print(f"Rozkład danych:\n{df['label'].value_counts()}")
    return df

def make_generators(df, mode='binary'):
    # Używamy stratify, aby utrzymać proporcje w zbiorze testowym
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    # Ważne: Zostawiamy PUSTE nawiasy (brak rescale), bo używamy EfficientNet!
    train_datagen = ImageDataGenerator(
        rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator()

    # Zmieniamy class_mode dla Etapu 1
    class_mode = "binary" if mode == 'binary' else "categorical"

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="filepath", y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode=class_mode, batch_size=BATCH_SIZE, shuffle=True
    )

    test_gen = test_datagen.flow_from_dataframe(
        test_df, x_col="filepath", y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode=class_mode, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_gen, test_gen