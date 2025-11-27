# train.py

from .data import load_dataframe, make_generators
from .model import build_model
from .callbacks import get_callbacks
from .plot import plot_history
from .config import EPOCHS, MODEL_OUTPUT_PATH


def train():
    df = load_dataframe()
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

    model.save(MODEL_OUTPUT_PATH)
    print(f"Model saved: {MODEL_OUTPUT_PATH}")
    plot_history(history)

    return history
