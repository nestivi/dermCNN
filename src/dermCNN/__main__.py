# src/dermCNN/__main__.py
from .train import train

if __name__ == "__main__":
    # Domyślnie uruchamiamy Etap 1 (binary)
    train(mode='binary')