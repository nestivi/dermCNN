# src/dermCNN/train.py

import os
import numpy as np
from sklearn.utils import class_weight  # <--- NOWY IMPORT
from .data import load_dataframe, make_generators
from .model import build_model
from .callbacks import get_callbacks
from .plot import plot_history
from .config import EPOCHS, MODEL_OUTPUT_PATH

def train():
    df = load_dataframe()

    if df.empty:
        raise ValueError("DataFrame is empty. Check CSV_PATH.")
    
    print(f"Loaded {len(df)} samples from dataset.")
    
    train_gen, test_gen = make_generators(df)

    # --- POPRAWIONA SEKCJA: OBLICZANIE WAG KLAS ---
    y_train = train_gen.classes 
    
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # NOWOŚĆ: Pierwiastkowanie wag, żeby nie były tak ekstremalne
    # Zmieni to wagę 13.0 na około 3.6, a 0.2 na 0.45.
    # To o wiele zdrowsze dla modelu.
    class_weights_vals = np.sqrt(class_weights_vals)
    
    class_weights = dict(enumerate(class_weights_vals))
    
    print("\nZłagodzone wagi klas (po pierwiastkowaniu):")
    for cls_idx, weight in class_weights.items():
        cls_name = list(train_gen.class_indices.keys())[list(train_gen.class_indices.values()).index(cls_idx)]
        print(f"  {cls_name}: {weight:.4f}")
    # ----------------------------------------

    model = build_model()
    # model.summary() # Możesz zakomentować, żeby nie śmiecić w logach

    callbacks = get_callbacks()

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights  # <--- PRZEKAZANIE WAG TUTAJ
    )

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    # Zapisz w nowym formacie .keras (zgodnie z ostrzeżeniem w logach)
    model.save(MODEL_OUTPUT_PATH.replace(".h5", ".keras")) 
    print(f"Model saved: {MODEL_OUTPUT_PATH.replace('.h5', '.keras')}")
    
    plot_history(history)

    return history