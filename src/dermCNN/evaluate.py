import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


try:
    from .data import load_dataframe
    from .config import IMG_SIZE, BATCH_SIZE, MODEL_OUTPUT_PATH
except ImportError:
    from data import load_dataframe
    from config import IMG_SIZE, BATCH_SIZE, MODEL_OUTPUT_PATH

def run_evaluation():
    # 1. Ładowanie danych i odtworzenie podziału (random_state=42 jest kluczowe!)
    print("--- Loading data ---")
    df = load_dataframe()
    _, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    print(f"Number of test images: {len(test_df)}")

    # 2. Generator walidacyjny BEZ TASOWANIA (shuffle=False)
    # Jest to niezbędne, aby predykcje pasowały do prawdziwych etykiet
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 3. Ładowanie modelu
    if not os.path.exists(MODEL_OUTPUT_PATH):
        print(f"Error: Model not found: {MODEL_OUTPUT_PATH}")
        return

    print(f"Loading model: {MODEL_OUTPUT_PATH}")
    model = load_model(MODEL_OUTPUT_PATH)

    # 4. Predykcja
    print("Generating predictions...")
    preds = model.predict(test_gen, verbose=1)
    
    # Zamiana prawdopodobieństw na klasy (0 lub 1)
    y_pred = (preds > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    # 5. Tworzenie macierzy
    cm = confusion_matrix(y_true, y_pred)
    # Pobranie nazw klas (zazwyczaj ['MEL', 'NV'] -> MEL=0, NV=1)
    class_names = list(test_gen.class_indices.keys())

    # 6. Rysowanie wykresu
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.title('Confusion Matrix')

    # Zapis do pliku
    save_dir = os.path.dirname(MODEL_OUTPUT_PATH)
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Plot saved: {save_path}")
    plt.show()

    # 7. Raport w konsoli (Precyzja, Czułość)
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    run_evaluation()