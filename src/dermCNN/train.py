# train.py

import os
from .data import load_dataframe, make_generators
from .model import build_model
from .callbacks import get_callbacks
from .plot import plot_history
from .config import EPOCHS, MODEL_OUTPUT_PATH


def train():
    df = load_dataframe()

    if df.empty:
        raise ValueError(
            "DataFrame is empty. Check CSV_PATH and BASE_DIR in config.py. "
            "Ensure the CSV exists and contains MEL/NV samples."
        )
    print(f"Loaded {len(df)} samples from dataset.")
    
    train_gen, test_gen = make_generators(df)

    model = build_model()
    model.summary()

    callbacks = get_callbacks()

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    model.save(MODEL_OUTPUT_PATH)
    print(f"Model saved: {MODEL_OUTPUT_PATH}")
    plot_history(history)

    return history
