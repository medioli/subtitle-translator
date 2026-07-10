"""
Traductor de subtítulos en tiempo real
========================================
Captura el audio que suena en tu computadora (de un video, stream, etc.),
lo transcribe con Whisper y lo traduce al idioma que elijas, mostrando
los subtítulos en una ventana flotante siempre-visible.

INSTALACIÓN
-----------
1. Python 3.9 - 3.11 (recomendado; versiones muy nuevas pueden dar problemas
   con algunas dependencias de audio).

2. Instalar dependencias:
   pip install faster-whisper argostranslate soundcard numpy

3. Audio del sistema (MUY IMPORTANTE, difiere por SO):

   - WINDOWS: no necesitás nada extra. `soundcard` usa WASAPI loopback
     para capturar directamente lo que está sonando.

   - LINUX: tampoco necesitás nada extra si usás PulseAudio/PipeWire
     (la mayoría de las distros modernas). El script busca automáticamente
     el "monitor" del dispositivo de salida.

   - macOS: Apple NO permite capturar el audio de salida directamente.
     Necesitás instalar un dispositivo de audio virtual gratuito:
       1. Instalá BlackHole (https://existential.audio/blackhole/)
       2. En "Configuración de Audio MIDI" de macOS, creá un
          "Dispositivo de Múltiples Salidas" que incluya tus parlantes
          + BlackHole 2ch, y seleccionalo como salida de audio.
       3. En este script, elegí "BlackHole 2ch" como DEVICE_NAME abajo.

USO
---
python subtitle_translator.py

La primera vez que corras el script, descarga el modelo de Whisper
(~150MB para "small") y los paquetes de idioma de Argos Translate
(pocos MB cada uno). Necesitás internet solo para esa primera descarga.
"""

import queue
import os
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass

import numpy as np
import soundcard as sc

# --- Soporte GPU (CUDA) en Windows ---
# Si instalaste las librerías CUDA vía pip (nvidia-cublas-cu12,
# nvidia-cudnn-cu12), sus DLLs quedan en site-packages/nvidia/*/bin
# y Windows no las encuentra automáticamente. Esto las agrega al PATH.
if sys.platform == "win32":
    try:
        import nvidia.cublas as _cublas_pkg
        import nvidia.cudnn as _cudnn_pkg

        cublas_bin = os.path.join(list(_cublas_pkg.__path__)[0], "bin")
        cudnn_bin = os.path.join(list(_cudnn_pkg.__path__)[0], "bin")

        for bin_path in (cublas_bin, cudnn_bin):
            if os.path.isdir(bin_path):
                os.add_dll_directory(bin_path)
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
    except ImportError:
        pass  # no hay GPU configurada, sigue en CPU sin problema

from faster_whisper import WhisperModel

# ============================================================
# CONFIGURACIÓN — Ajustá esto a tu gusto
# ============================================================

# Idioma de origen del audio (código ISO 639-1). "auto" para autodetectar.
SOURCE_LANG = "auto"

# Idioma al que querés traducir (código ISO 639-1: es, en, fr, pt, de, it...)
TARGET_LANG = "es"

# Nombre del dispositivo de audio a capturar. None = dispositivo de
# salida por defecto (recomendado para Windows/Linux).
# En macOS, poné el nombre exacto de tu dispositivo BlackHole, ej:
#   DEVICE_NAME = "BlackHole 2ch"
DEVICE_NAME = None

# Tamaño del modelo Whisper: tiny, base, small, medium, large-v3
# "small" es un buen balance velocidad/calidad para tiempo real en CPU.
# Si tenés GPU NVIDIA, "medium" o "large-v3" con device="cuda" da mejor calidad.
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cuda"          # "cuda" si tenés GPU NVIDIA con CUDA
WHISPER_COMPUTE_TYPE = "float16"   # "int8" es rápido en CPU; "float16" si usás GPU

# Duración de cada bloque de audio a transcribir (segundos).
# Más corto = más responsivo pero puede cortar palabras/frases.
# Con GPU podés usar bloques más chicos sin perder tanta precisión.
CHUNK_SECONDS = 3.5

# Motor de traducción:
#   "google"  -> usa Google Translate (gratis, necesita internet, MEJOR
#                calidad, recomendado para japonés/chino/coreano)
#   "offline" -> usa Argos Translate (100% local, sin internet, pero
#                con calidad más floja en varios idiomas, especialmente
#                japonés, donde traduce pasando primero por inglés)
TRANSLATION_ENGINE = "google"

# --- Ajustes para audio muy bajito (ASMR, susurros, voces suaves) ---

# Amplifica el audio antes de transcribir. 1.0 = sin cambios.
# Para ASMR/susurros probá 3.0 a 6.0. Si empieza a distorsionar
# o a "alucinar" texto que no existe, bajalo un poco.
AUDIO_GAIN = 4.0

# Sensibilidad del detector de voz (VAD). Va de 0.0 a 1.0.
# Más bajo = detecta voces más bajitas, pero puede confundir ruido
# de fondo con habla. Default de faster-whisper es 0.5.
VAD_THRESHOLD = 0.25

SAMPLE_RATE = 16000  # Whisper espera 16kHz

# ============================================================
# ESTADO COMPARTIDO ENTRE HILOS
# ============================================================

audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
text_queue: "queue.Queue[TranslatedLine]" = queue.Queue()


@dataclass
class TranslatedLine:
    original: str
    translated: str


# ============================================================
# 1) CAPTURA DE AUDIO
# ============================================================

def audio_capture_worker():
    """Captura audio del sistema en bloques y los pone en audio_queue."""
    if DEVICE_NAME:
        mic = sc.get_microphone(DEVICE_NAME, include_loopback=True)
    else:
        # Loopback del dispositivo de salida por defecto
        default_speaker = sc.default_speaker()
        mic = sc.get_microphone(default_speaker.name, include_loopback=True)

    print(f"[audio] Capturando desde: {mic.name}")

    with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
        while True:
            data = recorder.record(numframes=int(SAMPLE_RATE * CHUNK_SECONDS))
            # soundcard devuelve float32 en rango [-1, 1], shape (N, 1)
            mono = data[:, 0].astype(np.float32)
            mono = amplify_audio(mono, AUDIO_GAIN)
            audio_queue.put(mono)


def amplify_audio(chunk: np.ndarray, gain: float) -> np.ndarray:
    """Amplifica el audio y evita que se distorsione (clipping)."""
    if gain == 1.0:
        return chunk
    amplified = chunk * gain
    # Si algún pico se pasa de [-1, 1], lo comprime suavemente en vez
    # de cortarlo de golpe (evita ruido tipo "crack" por distorsión).
    peak = np.max(np.abs(amplified))
    if peak > 1.0:
        amplified = amplified / peak
    return amplified


# ============================================================
# 2) TRANSCRIPCIÓN (Speech-to-Text) con faster-whisper
# ============================================================

def transcription_worker():
    print("[stt] Cargando modelo Whisper... (puede tardar la primera vez)")
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    print("[stt] Modelo listo.")

    lang_arg = None if SOURCE_LANG == "auto" else SOURCE_LANG
    previous_text = ""  # guarda lo último transcripto, como contexto

    while True:
        chunk = audio_queue.get()
        segments, info = model.transcribe(
            chunk,
            language=lang_arg,
            vad_filter=True,  # descarta silencios automáticamente
            vad_parameters=dict(
                min_silence_duration_ms=300,
                threshold=VAD_THRESHOLD,
            ),
            beam_size=2,  # más bajo = más rápido (con GPU, 2 sigue siendo preciso)
            # Le pasamos lo último transcripto como pista de contexto,
            # así entiende mejor frases que quedaron cortadas entre bloques.
            initial_prompt=previous_text[-200:] if previous_text else None,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            previous_text = text
            translation_queue.put(text)


# ============================================================
# 3) TRADUCCIÓN — Google Translate (online, mejor calidad) o
#    Argos Translate (offline, más floja en algunos idiomas)
# ============================================================

translation_queue: "queue.Queue[str]" = queue.Queue()


def ensure_argos_language_pack(source: str, target: str):
    import argostranslate.package as package

    package.update_package_index()
    available = package.get_available_packages()
    match = next(
        (p for p in available if p.from_code == source and p.to_code == target),
        None,
    )
    if match is None:
        raise RuntimeError(
            f"No se encontró paquete de traducción {source} -> {target} en Argos Translate."
        )
    installed_codes = {
        (p.from_code, p.to_code) for p in package.get_installed_packages()
    }
    if (source, target) not in installed_codes:
        print(f"[translate] Descargando paquete de idioma {source} -> {target}...")
        path = match.download()
        package.install_from_path(path)


def translation_worker():
    if TRANSLATION_ENGINE == "google":
        from deep_translator import GoogleTranslator

        # GoogleTranslator soporta source="auto" de verdad (detecta el
        # idioma en cada frase), así que no hace falta un idioma fijo.
        src = SOURCE_LANG  # "auto" o un código como "ja", "en", etc.
        translator = GoogleTranslator(source=src, target=TARGET_LANG)
        print(f"[translate] Usando Google Translate (origen: {src}).")

        def do_translate(text: str) -> str:
            return translator.translate(text)

    else:
        # Argos Translate no soporta bien "auto"; si no especificaste
        # un idioma de origen, usa inglés como respaldo.
        src = SOURCE_LANG if SOURCE_LANG != "auto" else "en"
        import argostranslate.translate as argos_translate

        ensure_argos_language_pack(src, TARGET_LANG)
        print("[translate] Usando Argos Translate (offline).")

        def do_translate(text: str) -> str:
            return argos_translate.translate(text, src, TARGET_LANG)

    while True:
        text = translation_queue.get()
        try:
            translated = do_translate(text)
        except Exception as e:
            translated = f"[error de traducción: {e}]"
        text_queue.put(TranslatedLine(original=text, translated=translated))


# ============================================================
# 4) OVERLAY — ventana flotante con los subtítulos
# ============================================================

class SubtitleOverlay:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Subtítulos")
        root.attributes("-topmost", True)   # siempre visible
        root.overrideredirect(True)          # sin bordes de ventana
        root.attributes("-alpha", 0.85)      # semi-transparente

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        win_w, win_h = int(screen_w * 0.7), 110
        x = (screen_w - win_w) // 2
        y = int(screen_h * 0.85)
        root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        root.configure(bg="black")

        self.label = tk.Label(
            root,
            text="Esperando audio...",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="black",
            wraplength=win_w - 20,
            justify="center",
        )
        self.label.pack(expand=True, fill="both", padx=10, pady=10)

        # Permitir arrastrar la ventana con el mouse
        self.label.bind("<ButtonPress-1>", self._start_move)
        self.label.bind("<B1-Motion>", self._on_move)

        self._poll_queue()

    def _start_move(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_move(self, event):
        x = self.root.winfo_x() + (event.x - self._drag_x)
        y = self.root.winfo_y() + (event.y - self._drag_y)
        self.root.geometry(f"+{x}+{y}")

    def _poll_queue(self):
        try:
            while True:
                line = text_queue.get_nowait()
                self.label.config(text=line.translated)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)


# ============================================================
# MAIN
# ============================================================

def main():
    threading.Thread(target=audio_capture_worker, daemon=True).start()
    threading.Thread(target=transcription_worker, daemon=True).start()
    threading.Thread(target=translation_worker, daemon=True).start()

    root = tk.Tk()
    SubtitleOverlay(root)
    print("Overlay iniciado. Arrastrá la ventana para moverla. Cerrá con Ctrl+C en la terminal.")
    root.mainloop()


if __name__ == "__main__":
    main()
