import os
import cv2
import numpy as np

RUTA_ORIGEN = 'lsm'  # dataset original
RUTA_DESTINO = 'lsm2'  # nuevo dataset con imágenes originales + espejo

for clase in os.listdir(RUTA_ORIGEN):
    ruta_clase_origen = os.path.join(RUTA_ORIGEN, clase)
    ruta_clase_destino = os.path.join(RUTA_DESTINO, clase)
    os.makedirs(ruta_clase_destino, exist_ok=True)

    for img_name in os.listdir(ruta_clase_origen):
        img_path = os.path.join(ruta_clase_origen, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Guarda imagen original
        cv2.imwrite(os.path.join(ruta_clase_destino, img_name), img)

        # Crea y guarda imagen espejada
        espejo = cv2.flip(img, 1)
        nombre_espejo = f"mirror_{img_name}"
        cv2.imwrite(os.path.join(ruta_clase_destino, nombre_espejo), espejo)
