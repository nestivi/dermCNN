# src/dermCNN/train.py

import os
from .data import load_dataframe, make_generators
from .model import build_model
from .callbacks import get_callbacks
from .plot import plot_history
from .config import EPOCHS, MODEL_OUTPUT_PATH

def train(mode='binary'):
    """
    Uruchamia proces trenowania.
    Domyślnie mode='binary' trenuje model rozróżniający zmiany łagodne od złośliwych.
    """
    print(f"\n--- ROZPOCZĘCIE TRENOWANIA: ETAP 1 ({mode.upper()}) ---")
    
    # 1. Ładowanie danych (DataFrame) dla odpowiedniego trybu
    df = load_dataframe(mode=mode)

    if df.empty:
        raise ValueError("DataFrame is empty. Check CSV_PATH.")
    
    print(f"Załadowano {len(df)} próbek z datasetu.")
    
    # 2. Tworzenie generatorów z usuniętym skalowaniem (0-255 dla EfficientNet)
    train_gen, test_gen = make_generators(df, mode=mode)

    # 3. Budowa modelu (1 neuron wyjściowy z sigmoid dla 'binary')
    model = build_model(mode=mode)
    
    # Opcjonalnie: wyświetl podsumowanie modelu
    # model.summary()

    # 4. Inicjalizacja mechanizmów zwrotnych (np. EarlyStopping, Checkpoint)
    callbacks = get_callbacks()

    # UWAGA: Usunęliśmy wagi klas (class_weights), ponieważ przy podziale 
    # ~63% (łagodne) do ~37% (złośliwe) zbiór jest wystarczająco zbalansowany.

    # 5. Właściwa pętla treningowa
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 6. Zapisywanie wytrenowanego modelu w formacie .keras
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModel zapisany pomyślnie: {MODEL_OUTPUT_PATH}")
    
    # 7. Generowanie wykresów strat i celności
    plot_history(history)

    return history