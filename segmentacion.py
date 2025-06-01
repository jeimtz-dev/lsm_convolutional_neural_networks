import os
import cv2
from pathlib import Path
import mediapipe as mp

# Ruta base del dataset original y el nuevo destino para segmentadas
BASE_PATH = Path('./test/eval_v0')
OUTPUT_PATH = Path('./lsm_prueba')  # Puedes cambiar este nombre

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Función para segmentar la mano en la imagen
def segmentar_mano(imagen_bgr):
    h, w, _ = imagen_bgr.shape
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    resultado = hands.process(imagen_rgb)

    if not resultado.multi_hand_landmarks:
        return None  # No se detectó mano

    landmarks = resultado.multi_hand_landmarks[0].landmark
    x_coords = [int(lm.x * w) for lm in landmarks]
    y_coords = [int(lm.y * h) for lm in landmarks]
    x_min, x_max = max(min(x_coords) - 40, 0), min(max(x_coords) + 40, w)
    y_min, y_max = max(min(y_coords) - 40, 0), min(max(y_coords) + 60, h)

    return imagen_bgr[y_min:y_max, x_min:x_max]

# Crear el nuevo dataset con imágenes segmentadas
for carpeta_clase in sorted(BASE_PATH.iterdir()):
    if not carpeta_clase.is_dir():
        continue

    clase = carpeta_clase.name
    nueva_carpeta = OUTPUT_PATH / clase
    nueva_carpeta.mkdir(parents=True, exist_ok=True)

    for archivo in os.listdir(carpeta_clase):
        if not archivo.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        ruta_img = carpeta_clase / archivo
        imagen = cv2.imread(str(ruta_img))

        if imagen is None:
            print(f"[!] No se pudo leer la imagen: {ruta_img}")
            continue

        mano = segmentar_mano(imagen)
        if mano is None:
            print(f"[!] Mano no detectada en: {ruta_img}")
            continue

        # Guardar imagen segmentada (misma extensión que original)
        ruta_salida = nueva_carpeta / archivo
        cv2.imwrite(str(ruta_salida), mano)

print("\n✅ Segmentación completa. Imágenes guardadas en:", OUTPUT_PATH)
