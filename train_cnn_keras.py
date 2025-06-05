import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# Configuraciones
TAMANO_IMG = 200
RUTA_DATASET = "lsm_aument"
NUM_CLASES = 21
BATCH_SIZE = 32
EPOCHS = 30

datagen = ImageDataGenerator(
    rescale=1./255,              
    validation_split=0.20       
)

train_generator = datagen.flow_from_directory(
    RUTA_DATASET,
    target_size=(TAMANO_IMG, TAMANO_IMG),  # Redimensiona cada imagen
    color_mode='grayscale',                # Convierte a escala de grises (1 canal)
    batch_size=BATCH_SIZE,
    class_mode='categorical',              # Etiquetas codificadas como one-hot
    subset='training',                     # Usa el subconjunto de entrenamiento
    shuffle=True,                           # Mezcla aleatoriamente las imágenes
)

val_generator = datagen.flow_from_directory(
    RUTA_DATASET,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
)

modeloCNN2_AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),                                                                                 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASES, activation='softmax')
])

modeloCNN2_AD.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

inicio = time.time()
history = modeloCNN2_AD.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)

fin = time.time()
duracion = fin - inicio
print(f"Duración: {duracion:.2f} segundos")

modeloCNN2_AD.save("train_cnn_v2.h5")

plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.legend()
plt.show()
