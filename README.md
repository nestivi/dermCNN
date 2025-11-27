# dermCNN

## Description

Project Overview

This project uses Convolutional Neural Networks (CNNs) to classify dermatoscopic images of skin lesions. Currently, it supports binary classification between melanoma (MEL) and benign nevi (NV).

The project supports:

* Data loading and filtering for the ISIC 2019 dataset
* Train/test split
* Data augmentation
* Model training with callbacks
* Saving the trained model
* Plotting training progress

---

## Project Structure

```
bachelor/
├───results
│       .gitkeep
│
├───src
│   ├───dermCNN
│   │       callbacks.py
│   │       config.py
│   │       data.py
│   │       main.py
│   │       model.py
│   │       plot.py
│   │       train.py
│   │       train_old.py
│   │       __init__.py
│   │       __main__.py
│   │
│   └───dermCNN.egg-info
│           dependency_links.txt
│           PKG-INFO
│           requires.txt
│           SOURCES.txt
│           top_level.txt
│
├───tests
│       .gitkeep             
│
│   .gitignore
│   .python-version
│   pyproject.toml
└   README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nestivi/bachelor.git
cd dermCNN
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment:

* Windows:

```bash
.venv\Scripts\activate
```

* macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -e .
```

> **Note:** Make sure `tensorflow`, `pandas`, `scikit-learn`, and `matplotlib` are listed in pyproject.toml.

---

## Usage

### Running training:

```bash
python -m src.dermCNN
```

or using the CLI starter:

```bash
python src/dermCNN/main.py
```

The script will:

* Load and filter the dataset (MEL vs NV)
* Create train/test split
* Initialize the CNN model
* Train the model with data augmentation
* Save the trained model to `results/model_isic_cnn.h5`
* Plot training history (accuracy and loss)

---

## Tuning the Model

### Parameters to adjust:

* **Image size**: in `config.py` (`IMG_SIZE`)
* **Batch size**: in `config.py` (`BATCH_SIZE`)
* **Number of epochs**: in `train.py` (`epochs`)
* **Data augmentation**: in `train.py`, adjust `ImageDataGenerator` parameters:

  * `rotation_range`
  * `zoom_range`
  * `horizontal_flip`
* **CNN architecture**: in `model.py`:

  * number of layers
  * number of filters
  * kernel size
  * dense layer size

### Using Callbacks:

* **EarlyStopping**: stops training if validation loss stops improving
* **ModelCheckpoint**: saves the best model based on validation accuracy

Modify the parameters in `callbacks.py` to change behavior.

### Example: increase model complexity

```python
# model.py
layers.Conv2D(256, (3,3), activation='relu'),
layers.MaxPooling2D(),
```

### Example 2: lightweight model for faster testing
---
For quick testing of hyperparameter changes, you can train a smaller model with fewer layers or fewer filters, and fewer epochs:

In config.py, adjust parameters:

```python
IMG_SIZE = 128  # smaller images
BATCH_SIZE = 16
EPOCHS = 2      # quick test
```

In model.py, you can reduce the number of filters per layer for faster training.

Run the training as usual:

```bash
python -m src.dermCNN
```

This allows you to quickly verify changes in tuning without waiting for full training.

## Visualizing Training

The `plot.py` module can be used to visualize accuracy and loss per epoch:

```python
from plot import plot_training_history

plot_training_history(history)
```

This will generate plots for:

* Training vs validation accuracy
* Training vs validation loss

Plots are saved to `results/`.

---

## Notes

* Ensure the dataset path in `config.py` is correct.
* Training can take a long time depending on dataset size and GPU availability.
* Use `train_old.py` only for reference; the main pipeline is in `train.py`.

---

## License

This project is licensed under MIT License.