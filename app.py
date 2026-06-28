"""
app.py — Clasificación de emociones caninas en tiempo real.

Uso:
    python app.py

Muestra los modelos .keras disponibles en la carpeta models/,
te pide que elijas uno y arranca la cámara. Presioná 'q' para salir.
"""

import os
import sys
import glob

import cv2
import numpy as np

# ── Suprimir logs verbosos de TensorFlow antes de importarlo ──────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras


# ── Constantes ─────────────────────────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
IMG_SIZE     = (299, 299)          # InceptionV3 espera 299×299
CLASSES      = ["angry", "happy", "relaxed", "sad"]
CONF_THRESH  = 0.50                # confianza mínima para mostrar la etiqueta

# Paleta de colores por clase (BGR para OpenCV)
CLASS_COLORS = {
    "angry":   (0,   50,  220),   # rojo
    "happy":   (0,  200,   50),   # verde
    "relaxed": (200, 150,   0),   # azul‑celeste
    "sad":     (180,  80, 180),   # violeta
}

# ── Buscar modelos ──────────────────────────────────────────────────────────────
def list_models() -> list[str]:
    """Devuelve las rutas a los archivos .keras dentro de models/."""
    pattern = os.path.join(MODELS_DIR, "*.keras")
    return sorted(glob.glob(pattern))


def choose_model(models: list[str]) -> str:
    """Muestra el menú de selección y devuelve la ruta elegida."""
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Clasificador de Emociones Caninas — InceptionV3   ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    if not models:
        print(f"[ERROR] No se encontraron modelos .keras en: {MODELS_DIR}")
        print("        Corré el notebook primero para generar el modelo.")
        sys.exit(1)

    print("Modelos disponibles:")
    for i, path in enumerate(models, start=1):
        name = os.path.basename(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [{i}] {name}  ({size_mb:.1f} MB)")

    print()
    while True:
        raw = input(f"Elegí un modelo [1-{len(models)}]: ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]
        print(f"    → Ingresá un número entre 1 y {len(models)}.")


# ── Preprocesamiento ────────────────────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convierte un frame BGR de OpenCV al tensor que espera InceptionV3."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    tensor  = tf.keras.applications.inception_v3.preprocess_input(
        resized.astype("float32")
    )
    return np.expand_dims(tensor, axis=0)   # (1, 299, 299, 3)


# ── Overlay de resultados ───────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, label: str, confidence: float) -> np.ndarray:
    """
    Dibuja un recuadro semitransparente y la predicción sobre el frame.
    Si la confianza es baja, muestra "?" en gris.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if confidence >= CONF_THRESH:
        color = CLASS_COLORS.get(label, (255, 255, 255))
        text  = f"{label.upper()}  {confidence*100:.1f}%"
    else:
        color = (120, 120, 120)
        text  = f"?  {confidence*100:.1f}%"

    # Barra de fondo semitransparente
    bar_h = 70
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), color, -1)
    alpha = 0.55
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Texto principal
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.6
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = h - bar_h + th + 14

    # Sombra
    cv2.putText(frame, text, (tx + 2, ty + 2), font, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Texto
    cv2.putText(frame, text, (tx, ty), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    # Barra de confianza (parte superior del recuadro)
    bar_w = int(w * confidence)
    cv2.rectangle(frame, (0, h - bar_h - 6), (bar_w, h - bar_h), color, -1)

    # Leyenda 'q' para salir
    cv2.putText(frame, "Presiona 'q' para salir", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


# ── Bucle principal ─────────────────────────────────────────────────────────────
def run(model_path: str) -> None:
    print(f"\nCargando modelo: {os.path.basename(model_path)} …")
    model = keras.models.load_model(model_path)
    print("Modelo cargado correctamente.")

    # Calentamiento: una pasada en blanco para compilar grafos de TF
    _ = model.predict(np.zeros((1, *IMG_SIZE, 3), dtype="float32"), verbose=0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara (índice 0).")
        sys.exit(1)

    # Ajustes de captura
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    window_name = "Dog Emotion Classifier — TP Final IIA"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[INFO] Cámara activa. Mostrá un perro frente a la cámara.")
    print("       Presioná 'q' o Esc para salir.\n")

    PREDICT_EVERY = 3   # predecir cada N frames (balance velocidad/fluidez)
    frame_count   = 0
    label         = "?"
    confidence    = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No se pudo leer el frame. Reintentando…")
            continue

        frame_count += 1

        if frame_count % PREDICT_EVERY == 0:
            tensor      = preprocess_frame(frame)
            probs       = model.predict(tensor, verbose=0)[0]   # shape (4,)
            idx         = int(np.argmax(probs))
            label       = CLASSES[idx]
            confidence  = float(probs[idx])

        display = draw_overlay(frame, label, confidence)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # 'q' o Esc
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nApp cerrada. ¡Hasta luego!")


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    models  = list_models()
    chosen  = choose_model(models)
    run(chosen)
