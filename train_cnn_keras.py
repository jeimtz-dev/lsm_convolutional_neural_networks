import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
import time

# Configuraciones
TAMANO_IMG = 200
RUTA_DATASET = "lsm"
NUM_CLASES = 21
BATCH_SIZE = 64
EPOCHS =20


# Generador con aumento de datos para entrenamiento
datagen = ImageDataGenerator(
    rescale=1./255,              # Normaliza los pixeles al rango [0,1]
    rotation_range=30,           # Rotación aleatoria de hasta 30°
    width_shift_range=0.2,       # Desplazamiento horizontal aleatorio
    height_shift_range=0.2,      # Desplazamiento vertical aleatorio
    shear_range=15,              # Cizallamiento aleatorio
    zoom_range=[0.7, 1.4],       # Zoom aleatorio
    horizontal_flip=True,        # Voltea horizontalmente
    validation_split=0.30       # Separa un 30% de los datos para validación
)

# Generador de datos de entrenamiento (85% del total)
train_generator = datagen.flow_from_directory(
    RUTA_DATASET,
    target_size=(TAMANO_IMG, TAMANO_IMG),  # Redimensiona cada imagen
    color_mode='grayscale',                # Convierte a escala de grises (1 canal)
    batch_size=BATCH_SIZE,
    class_mode='categorical',              # Etiquetas codificadas como one-hot
    subset='training',                     # Usa el subconjunto de entrenamiento
    shuffle=True                           # Mezcla aleatoriamente las imágenes
)

# Generador de datos de validación (15% del total)
val_generator = datagen.flow_from_directory(
    RUTA_DATASET,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

modeloCNN2_AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASES, activation='softmax')
])

modeloCNN2_AD.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
tensorboard = TensorBoard(log_dir='CNN_KERAS_V2_LOGS/CNN_KERAS')

inicio = time.time()
modeloCNN2_AD.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=[tensorboard]
)
fin = time.time()
duracion = fin - inicio
print("Duracion: "+ duracion)
# Guardar modelo
modeloCNN2_AD.save("CNN_KERAS_v2.h5")
