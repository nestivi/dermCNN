# src/dermCNN/__main__.py
import argparse
from .train import train

if __name__ == "__main__":
    # Tworzymy parser argumentów
    parser = argparse.ArgumentParser(description="Trening systemu kaskadowego DermCNN.")
    
    # Dodajemy opcję --mode
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['binary', 'malignant_only'], 
        default='binary', # Domyślnie uruchomi się Etap 1
        help="Wybierz tryb trenowania: 'binary' (łagodne vs złośliwe) lub 'malignant_only' (klasyfikacja nowotworów)."
    )
    
    # Pobieramy argumenty z terminala
    args = parser.parse_args()
    
    # Uruchamiamy trenowanie z wybranym argumentem
    train(mode=args.mode)