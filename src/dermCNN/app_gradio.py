# app_gradio.py

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- KONFIGURACJA ---
MODEL_STAGE1_PATH = os.path.join("results", "model_stage1_binary.keras")
MODEL_STAGE2_PATH = os.path.join("results", "model_stage2_malignant.keras")
IMG_SIZE = 224

# Nazwy klas z Etapu 2 (alfabetycznie, tak jak ułożył je Keras podczas treningu)
STAGE2_CLASSES = ['AK (Rogowacenie słoneczne)', 
                  'BCC (Rak podstawnokomórkowy)', 
                  'MEL (Czerniak)', 
                  'SCC (Rak płaskonabłonkowy)']

# Zmienne na modele, aby załadować je tylko raz przy starcie aplikacji
model_stage1 = None
model_stage2 = None

def load_models():
    """Funkcja ładująca oba modele do pamięci operacyjnej."""
    global model_stage1, model_stage2
    if model_stage1 is None or model_stage2 is None:
        print("Ładowanie modeli do pamięci... To może chwilę potrwać.")
        model_stage1 = tf.keras.models.load_model(MODEL_STAGE1_PATH)
        model_stage2 = tf.keras.models.load_model(MODEL_STAGE2_PATH)
        print("Modele załadowane pomyślnie!")

def predict_pipeline(image):
    """
    Główna funkcja kaskadowa przetwarzająca wgrane zdjęcie.
    """
    if image is None:
        return None
    
    # Upewniamy się, że modele są załadowane
    try:
        load_models()
    except Exception as e:
        return {f"Błąd ładowania modeli: {e}": 1.0}

    # 1. Przetwarzanie obrazu pod modele EfficientNet
    # Zmieniamy rozmiar i tworzymy batch o rozmiarze 1
    # Pamiętaj: NIE skalujemy przez 255.0, bo to zepsuje predykcję!
    img_resized = Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # 2. ETAP 1: Czy to jest groźne?
    # sigmoid zwraca jedną wartość od 0.0 (benign) do 1.0 (malignant)
    pred_stage1 = model_stage1.predict(img_array)[0][0]
    prob_malignant = float(pred_stage1)
    prob_benign = 1.0 - prob_malignant

    results = {}

    # Decyzja kaskadowa (ustalamy próg na 50%)
    if prob_malignant < 0.5:
        # Jeśli zmiana jest łagodna, kończymy diagnozę
        results["🟢 Zmiana ŁAGODNA (Benign)"] = prob_benign
        results["Ryzyko złośliwości"] = prob_malignant
        return results
    else:
        # 3. ETAP 2: Co to za nowotwór?
        results["🔴 ZMIANA ZŁOŚLIWA - Prawdopodobieństwo"] = prob_malignant
        
        # Przekazujemy zdjęcie do drugiego modelu
        pred_stage2 = model_stage2.predict(img_array)[0]
        
        # Zapisujemy wyniki drugiego etapu do słownika
        for i, class_name in enumerate(STAGE2_CLASSES):
            # Dodajemy wcięcie dla lepszej czytelności w GUI
            results[f"   ↳ {class_name}"] = float(pred_stage2[i])
            
        return results

# --- INTERFEJS UŻYTKOWNIKA ---
# Konstruujemy widok strony WWW
interface = gr.Interface(
    fn=predict_pipeline,
    inputs=gr.Image(label="Wgraj zdjęcie dermatoskopowe zmiany skórnej"),
    outputs=gr.Label(num_top_classes=5, label="Wynik diagnozy AI"),
    title="DermCNN AI - Kaskadowy System Klasyfikacji",
    description="Algorytm najpierw ocenia czy zmiana jest złośliwa. Jeśli tak, uruchamia drugi moduł w celu identyfikacji konkretnego typu nowotworu."
)

if __name__ == "__main__":
    # Uruchomienie lokalnego serwera WWW
    interface.launch()