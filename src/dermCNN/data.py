# data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import BASE_DIR, CSV_PATH, IMG_SIZE, BATCH_SIZE, CLASSES

def load_dataframe():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    
    # ISIC 2019 ma kolumny dla każdej klasy (One-Hot). 
    # Tworzymy jedną kolumnę 'label', wybierając nazwę kolumny, która ma wartość 1.
    # axis=1 oznacza operację wiersz po wierszu.
    df["label"] = df[CLASSES].idxmax(axis=1)

    # Dodajemy pełną ścieżkę
    df["filepath"] = df["image"].apply(lambda x: os.path.join(BASE_DIR, x + ".jpg"))
    
    # Filtrujemy tylko istniejące pliki (dobre zabezpieczenie)
    df = df[df["filepath"].apply(os.path.exists)]
    
    print(f"Data Loaded. Distribution:\n{df['label'].value_counts()}")
    return df

def make_generators(df):
    # Stratify jest kluczowe przy niezbalansowanych danych!
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2,
        stratify=df["label"], 
        random_state=42
    )

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,      # Zwiększyłem augmentację
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,     # Zmiany skórne nie mają orientacji góra/dół
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="categorical",  # Zmiana na categorical dla >2 klas
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False # Ważne: nie tasujemy walidacji, żeby wyniki były powtarzalne
    )

    return train_gen, test_gen