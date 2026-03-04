# src/dermCNN/evaluate.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import argparse

from .data import load_dataframe, make_generators
from .config import MODEL_OUTPUT_PATH_STAGE1, MODEL_OUTPUT_PATH_STAGE2

def evaluate_model(mode='binary'):
    print(f"\n--- ROZPOCZĘCIE EWALUACJI: ETAP ({mode.upper()}) ---")

    # 1. Wybór odpowiedniej ścieżki modelu w zależności od trybu
    if mode == 'binary':
        model_path = MODEL_OUTPUT_PATH_STAGE1
    else:
        model_path = MODEL_OUTPUT_PATH_STAGE2

    if not os.path.exists(model_path):
        print(f"Błąd: Nie znaleziono pliku modelu {model_path}.")
        return

    # 2. Załadowanie danych (używamy tej samej logiki, żeby podział na testowy był identyczny)
    df = load_dataframe(mode=mode)
    
    # Rozpakowujemy generatory, ale interesuje nas tylko test_gen
    _, test_gen = make_generators(df, mode=mode) 

    # 3. Załadowanie modelu do pamięci
    print("Ładowanie modelu...")
    model = tf.keras.models.load_model(model_path)

    # 4. Generowanie predykcji dla całego zbioru testowego
    print("Trwa ocenianie zdjęć testowych... To potrwa chwilę.")
    predictions = model.predict(test_gen)

    # 5. Przetworzenie wyników
    y_true = test_gen.classes # Prawdziwe etykiety (z folderów/csv)
    class_labels = list(test_gen.class_indices.keys()) # Nazwy klas (np. ['benign', 'malignant'])

    if mode == 'binary':
        # Zmiana prawdopodobieństwa z sigmoid na klasy 0 i 1 (próg 50%)
        y_pred = (predictions > 0.5).astype(int).flatten()
    else:
        # Zmiana z softmax - wybieramy klasę z najwyższym wynikiem
        y_pred = np.argmax(predictions, axis=1)

    # 6. Rysowanie Macierzy Pomyłek (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    # sns.heatmap tworzy piękny, kolorowy wykres
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Macierz Pomyłek AI - {mode.upper()}')
    plt.ylabel('Rzeczywista diagnoza')
    plt.xlabel('Predykcja modelu')
    
    cm_path = os.path.join("results", f"confusion_matrix_{mode}.png")
    plt.savefig(cm_path)
    print(f"Zapisano wykres macierzy pomyłek: {cm_path}")
    
    # 7. Wyświetlenie okienka z wykresem
    plt.show()

    # 8. Generowanie raportu tekstowego (Czułość, Precyzja)
    print("\n--- RAPORT KLASYFIKACJI ---")
    print("Skopiuj to do swojej pracy licencjackiej!")
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)
    
    # Zapis raportu do pliku tekstowego na później
    report_path = os.path.join("results", f"classification_report_{mode}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Zapisano raport tekstowy do: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ewaluacja modelu DermCNN.")
    parser.add_argument('--mode', type=str, choices=['binary', 'malignant_only'], default='binary')
    args = parser.parse_args()
    
    evaluate_model(mode=args.mode)