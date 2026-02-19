# model.py
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from .config import IMG_SIZE, NUM_CLASSES

def build_model():
    # Pobieramy pretrenowany model (bez górnych warstw klasyfikujących)
    base_model = applications.EfficientNetB0(
        weights='imagenet',  # Wagi nauczone na ImageNet
        include_top=False,   # Odrzucamy ostatnią warstwę 1000 klas ImageNet
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Zamrażamy wagi bazowe - na początku uczymy tylko naszą nową "głowę"
    base_model.trainable = False 

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), # Zamienia mapy cech na wektor
        layers.BatchNormalization(),
        layers.Dropout(0.3),             # Zapobiega overfittingowi
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax') # Softmax dla wielu klas
    ])

    # Learning rate - dla transfer learningu często lepiej zacząć delikatnie
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy", # Musi być categorical dla >2 klas
        metrics=["accuracy"]
    )

    return model