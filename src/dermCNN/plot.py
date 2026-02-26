# src/dermCNN/plot.py

import matplotlib.pyplot as plt

# Dodaliśmy parametr mode='binary' (jako wartość domyślna)
def plot_history(history, mode='binary'):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    # Tytuł wykresu będzie dynamiczny!
    plt.title(f"Accuracy ({mode})")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    # Tytuł wykresu będzie dynamiczny!
    plt.title(f"Loss ({mode})")
    plt.legend()

    # Dynamiczna nazwa pliku - już nic się nie nadpisze
    plt.savefig(f"results/training_plot_{mode}.png")
    plt.show()