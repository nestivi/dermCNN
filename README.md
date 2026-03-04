# DermCNN AI - Cascade Classification System

## Description

Project Overview

This project implements an advanced Cascade Convolutional Neural Network (CNN) system for diagnosing dermatoscopic images of skin lesions, developed as part of a Bachelor's thesis. 

Instead of a simple binary classifier, it utilizes a **Two-Stage Cascade Architecture** powered by Transfer Learning (EfficientNetB0):
1. **Stage 1 (Binary):** Determines if a lesion is Benign (NV, BKL, DF, VASC) or Malignant (MEL, BCC, AK, SCC).
2. **Stage 2 (Multi-class):** If malignant, it classifies the specific type of skin cancer.



The project supports:
* Automated data loading and preprocessing from the ISIC 2019 dataset.
* Stratified train/test splitting to maintain class balance.
* Image data augmentation.
* Transfer Learning using frozen ImageNet weights.
* Comprehensive evaluation (Confusion Matrices, Classification Reports).
* A user-friendly Web UI built with Gradio.

---

## Project Structure

```text
bachelor/
├───results/                 # Saved models (.keras), plots, and logs
│
├───src/
│   ├───dermCNN/
│   │       __init__.py
│   │       __main__.py      # CLI Entry point for training
│   │       config.py        # Global settings & hyperparameters
│   │       data.py          # Data pipeline & augmentation
│   │       model.py         # EfficientNetB0 architecture
│   │       train.py         # Training pipeline orchestration
│   │       callbacks.py     # EarlyStopping, ModelCheckpoint
│   │       plot.py          # Training history visualization
│   │       evaluate.py      # Testing & Confusion Matrix generation
│   │       app_gradio.py    # Web interface for AI diagnosis
│   │
│   └───dermCNN.egg-info/
│
├───tests/                   # ISIC_2019_Training_Input & GroundTruth.csv
│   ├───ISIC_2019Training_Input/                # Dermatoscopic images of skin lesions
│   └───ISIC_2019_Training_GroundTruth.csv
│
│   .gitignore
│   .python-version
│   pyproject.toml
└── README.md
```

---

## Installation

1. Clone the repository:
  ```bash
  git clone [https://github.com/nestivi/bachelor.git](https://github.com/nestivi/bachelor.git)
  cd bachelor
  ```

2. Create a virtual environment:
  ```bash
  python -m venv .venv
  ```

3. Activate the virtual environment:
  * Windows:
    ```bash
    .venv/Scripts/activate
    ```

  * macOS/Linux:
    ```bash
    source .venv/bin/activate
    ```

4. Install the package:
  ```bash
  pip install -e .
  ```

---

## Usage

  1. Training the models

    You can train the models directly using the CLI module. It accepts a --mode argument to specify which stage of the cascade to train.

    Train Stage 1 (Benign vs. Malignant):
      ```bash
      python -m src.dermCNN --mode binary
      ```

    Train Stage 2 (Malignant tumor types):
      ```bash
      python -m src.dermCNN --mode malignant_only
      ```

    The script will automatically:
      - Load the dataset.
      - Train the EfficientNetB0-based mdoel.
      - Save the best model to results/best_model_<mode>.keras.
      - Generate training history plots.

  2. Evaluating the Models

    To evaluate the trained models on the test set and generate scientific metrics (Precision, Recall, F1-Score) along with a Seaborn Confusion Matrix:
      ```bash
      python -m src.dermCNN.evaluate --mode binary
      ```
      ```bash
      python -m src.dermCNN.evaluate --mode malignant_only
      ```
    Results (images and .txt reports) will be saved in the results/ folder.

  3. Launching the Web UI
    To use the system interactively, launch the Gradio web interface. The app lazily loads both trained models and performs a full cascade diagnosis on uploaded images.

      ```bash
      python src/dermCNN/app_gradio.py
      ```

    Open the provided local URL in your web browser.

---

## Tuning the Model

  Hyperparameters
    Centralized configurations can be found in src/dermCNN/config.py:

      - IMG_SIZE = 224 (Required by EfficientNetB0)
      - BATCH_SIZE = 32
      - EPOCHS = 20

  Architecture & Transfer Learning

    The network architecture is defined in src/dermCNN/model.py. The base EfficientNetB0 model is frozen to utilize pre-trained ImageNet features.
    If you want to fine-tune the model or increase its complexity, you can:

      - Modify the Dense layers in the classification head.
      - Adjust the Dropout rate (currently 0.3) to prevent overfitting.
      - Unfreeze the top layers of the base model for fine-tuning.

  Callbacks

    Training is monitored via src/dermCNN/callbacks.py.

      - EarlyStopping: Stops training if val_loss doesn't improve for 5 epochs.
      - ModelCheckpoint: Saves only the best performing weights.

---

Visualizing Results

  The pipeline automatically handles plotting.

    - Training History: plot.py saves training_plot_binary.png showing Accuracy and Loss curves.
    - Evaluation: evaluate.py generates high-quality Confusion Matrices suitable for academic papers.

  ---

## License

This project is licensed under the MIT License.