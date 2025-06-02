import tensorflow as tf

# Cargar modelo entrenado
modelo = tf.keras.models.load_model('train_cnn_v1.h5')

# Convertir a TFLite
convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
modelo_tflite = convertidor.convert()

# Guardar
with open("train_cnn_v1.tflite", "wb") as f:
    f.write(modelo_tflite)
