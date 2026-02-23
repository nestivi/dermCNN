# src/dermCNN/model.py
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from .config import IMG_SIZE

def build_model(mode='binary'):
    base_model = applications.EfficientNetB0(
        weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False 

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3)
    ])

    # Dodajemy odpowiednią "głowę" w zależności od etapu
    if mode == 'binary':
        # 1 neuron, funkcja sigmoid (0.0 = benign, 1.0 = malignant)
        model.add(layers.Dense(1, activation='sigmoid'))
        loss_fn = "binary_crossentropy"
    else:
        # Etap 2 (dla 4 klas złośliwych)
        model.add(layers.Dense(4, activation='softmax'))
        loss_fn = "categorical_crossentropy"

    # Używamy wolniejszego uczenia, które ustawiliśmy poprzednio
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    return model