# main.py

import sys
from .train import train

def main():
    """
    Główna funkcja uruchamiająca proces trenowania modelu.
    """
    print("--- Rozpoczęcie trenowania modelu dermCNN ---")
    try:
        train()
        print("--- Trenowanie modelu zakończone pomyślnie ---")
    except Exception as e:
        print(f"Błąd podczas trenowania: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()