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
│   └───    app_gradio.py    # Web interface for AI diagnosis
│
├───tests/                   # ISIC_2019_Training_Input & GroundTruth.csv
│   ├───ISIC_2019Training_Input/                # Dermatoscopic images of skin lesions
│   └───ISIC_2019_Training_GroundTruth.csv
│
│   .gitignore
│   pyproject.toml
└── README.md
```

---

## Dataset: ISIC Challenge 2019
Link to download datasets:
https://challenge.isic-archive.com/data/#2019

* **IMPORTANT**: to train your own model, your train images must be in folder: **tests/**


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

This repository uses a dual-licensing approach to respect the strict non-commercial terms of the medical dataset used for training:

* **Source Code:** All original source code in this repository is licensed under the [MIT License](LICENSE).
* **Trained Models (Weights):** The pre-trained models/weights provided in this repository were trained on the ISIC 2019 dataset. As derivative works, they are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license. **You may not use these models for commercial purposes.**
* **Ground Truth Data:** The ground truth files included here are part of the original ISIC 2019 dataset and are strictly licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

By using the models or the data provided in this repository, you agree to comply with the non-commercial restrictions set by the original dataset authors.

---

## Acknowledgements and Citations

The dataset used in this project is the **ISIC 2019 Challenge Dataset** (licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)). To comply with the attribution requirements, the aggregate "ISIC 2019" data is credited to the following sources:

* **BCN_20000 Dataset**: (c) Department of Dermatology, Hospital Clínic de Barcelona

* **HAM10000 Dataset**: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161

* **MSK Dataset: (c) Anonymous**; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368 

**Relevant Publications:**
* [1] Tschandl P., Rosendahl C. & Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi.10.1038/sdata.2018.161 (2018)

* [2] Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)", 2017; arXiv:1710.05006.

* [3] Hernández-Pérez C, Combalia M, Podlipnik S, Codella NC, Rotemberg V, Halpern AC, Reiter O, Carrera C, Barreiro A, Helba B, Puig S, Vilaplana V, Malvehy J. BCN20000: Dermoscopic lesions in the wild. Scientific Data. 2024 Jun 17;11(1):641.

---
