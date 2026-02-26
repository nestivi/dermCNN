# src/dermCNN/train.py

import os
from .data import load_dataframe, make_generators
from .model import build_model
from .callbacks import get_callbacks
from .plot import plot_history
from .config import EPOCHS, MODEL_OUTPUT_PATH_STAGE1, MODEL_OUTPUT_PATH_STAGE2

def train(mode='binary'):
    print(f"\n--- ROZPOCZĘCIE TRENOWANIA: ({mode.upper()}) ---")
    
    # Wybór odpowiedniej ścieżki zapisu w zależności od trybu
    if mode == 'binary':
        output_path = MODEL_OUTPUT_PATH_STAGE1
    else:
        output_path = MODEL_OUTPUT_PATH_STAGE2
    
    df = load_dataframe(mode=mode)

    if df.empty:
        raise ValueError("DataFrame is empty. Check CSV_PATH.")
    
    print(f"Załadowano {len(df)} próbek z datasetu.")
    
    train_gen, test_gen = make_generators(df, mode=mode)
    model = build_model(mode=mode)
    
    # Przekazujemy 'mode' do callbacków, aby nie nadpisywały checkpointów
    callbacks = get_callbacks(mode=mode)

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Zapis korzysta z dynamicznie wybranej ścieżki (output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"\nModel zapisany pomyślnie: {output_path}")
    
    # Przekazujemy 'mode' do plotów, by zapisać wykres z odpowiednią nazwą
    plot_history(history, mode=mode)

    return history