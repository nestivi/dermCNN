# app_gradio.py
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from .config import MODEL_OUTPUT_PATH_STAGE1, MODEL_OUTPUT_PATH_STAGE2, IMG_SIZE


# Ścieżki do obu modeli
MODEL_OUTPUT_PATH_STAGE1 = os.path.join("results", "model_stage1_binary.keras")
MODEL_OUTPUT_PATH_STAGE2 = os.path.join("results", "model_stage2_malignant.keras")
IMG_SIZE = 224

# Keras automatycznie sortuje klasy alfabetycznie podczas treningu
STAGE2_CLASSES = ['AK (Rogowacenie)', 'BCC (Rak podstawnokomórkowy)', 'MEL (Czerniak)', 'SCC (Rak płaskonabłonkowy)']

# Zmienne globalne na modele (aby załadować je tylko raz)
model_stage1 = None
model_stage2 = None

def load_models():
    global model_stage1, model_stage2
    if model_stage1 is None or model_stage2 is None:
        print("Ładowanie modeli...")
        model_stage1 = tf.keras.models.load_model(MODEL_OUTPUT_PATH_STAGE1)
        model_stage2 = tf.keras.models.load_model(MODEL_OUTPUT_PATH_STAGE2)
        print("Modele załadowane pomyślnie!")

def predict_pipeline(image):
    if image is None:
        return None
    
    try:
        load_models()
    except Exception as e:
        return {f"Błąd ładowania modeli: Upewnij się, że masz oba pliki .keras w folderze results. ({e})": 1.0}

    # Przygotowanie obrazu (bez /255, bo to EfficientNet)
    img_resized = Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # ETAP 1: Klasyfikacja binarna
    # Wynik z Sigmoid: 0.0 do 1.0. Keras układa: benign(0), malignant(1)
    pred_stage1 = model_stage1.predict(img_array)[0][0]
    prob_malignant = float(pred_stage1)
    prob_benign = 1.0 - prob_malignant

    # Jeśli model uzna, że zmiana jest łagodna (np. > 50%)
    if prob_benign > 0.5:
        return {
            "Wynik: Zmiana ŁAGODNA (Benign)": prob_benign,
            "Ryzyko złośliwości": prob_malignant
        }
    
    # ETAP 2: Jeśli zmiana jest złośliwa (Malignant > 50%)
    pred_stage2 = model_stage2.predict(img_array)[0]
    
    results = {"⚠️ ZMIANA ZŁOŚLIWA - Typ:" : prob_malignant}
    
    # Dodajemy wyniki z drugiego modelu
    for i, class_name in enumerate(STAGE2_CLASSES):
        # Mnożymy przez prawdopodobieństwo z etapu 1, by zachować skalę, 
        # lub zwracamy czyste prawdopodobieństwo z etapu 2. Zwróćmy czyste dla czytelności:
        results[f"   ↳ {class_name}"] = float(pred_stage2[i])
        
    return results

# Interfejs Gradio
interface = gr.Interface(
    fn=predict_pipeline,
    inputs=gr.Image(label="Wgraj zdjęcie dermatoskopowe"),
    outputs=gr.Label(num_top_classes=5, label="Werdykt AI"),
    title="AI Dermatoskop - System Kaskadowy",
    description="System dwuetapowy. Najpierw ocenia czy zmiana jest łagodna, czy złośliwa. Jeśli wykryje zagrożenie, klasyfikuje konkretny typ nowotworu (Czerniak, Rak podstawnokomórkowy, etc.)."
)

if __name__ == "__main__":
    interface.launch()