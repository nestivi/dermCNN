import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import tensorflow as tf

def train():
    # -----------------------------
    # 1. Konfiguracja
    # -----------------------------
    BASE_DIR = r"C:\Martin\Studia\Praca Licencjacka\test\tests\ISIC_2019_Training_Input\ISIC_2019_Training_Input"
    CSV_PATH = r"C:\Martin\Studia\Praca Licencjacka\test\tests\ISIC_2019_Training_GroundTruth.csv"

    IMG_SIZE = 224
    BATCH_SIZE = 32

    # -----------------------------
    # 2. Wczytanie i filtrowanie CSV
    # -----------------------------
    df = pd.read_csv(CSV_PATH)

    # Bierzemy 2 klasy
    df = df[(df['MEL'] == 1) | (df['NV'] == 1)]

    # Ustawiamy etykiety
    df['label'] = df['MEL'].apply(lambda x: "MEL" if x == 1 else "NV")

    # 🔥 DODAJEMY POPRAWNE TWORZENIE ŚCIEŻEK 🔥
    df["filepath"] = df["image"].apply(lambda x: os.path.join(BASE_DIR, x + ".jpg"))

    # 🔥 DEBUG — sprawdźmy czy pliki istnieją 🔥
    print("=== DEBUG CHECK ===")
    print(df.head())
    print("Example filepaths:", df["filepath"].head().tolist())

    exists = df["filepath"].apply(os.path.exists).sum()
    print(f"Plików istnieje: {exists} / {len(df)}")

    missing = df[~df["filepath"].apply(os.path.exists)]
    print("Brakujące pierwsze 5:")
    print(missing.head())
    print("===================")

    # Usuwamy rekordy bez obrazów
    df = df[df["filepath"].apply(os.path.exists)]


    # -----------------------------
    # 3. Podział train/test
    # -----------------------------
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # -----------------------------
    # 4. Generatory danych
    # -----------------------------
    train_datagen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        rescale=1/255.0
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="binary",
        batch_size=BATCH_SIZE
    )

    # -----------------------------
    # 5. Model CNN
    # -----------------------------
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')   # binary classification
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.summary()

    # -----------------------------
    # 6. Trenowanie
    # -----------------------------
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10
    )

    # -----------------------------
    # 7. Zapis modelu
    # -----------------------------
    model.save("tests/model_isic_cnn.h5")
    print("Model zapisany jako model_isic_cnn.h5")
