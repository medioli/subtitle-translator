# Traductor de Subtítulos en Tiempo Real

Captura el audio que suena en tu computadora (video, stream, videollamada,
etc.), lo transcribe con IA y lo traduce en vivo, mostrando los subtítulos
en una ventana flotante siempre-visible sobre cualquier programa.

## Características

- 🎧 Captura el audio del sistema (no el micrófono) — funciona con
  cualquier reproductor de video, navegador o app.
- 🗣️ Transcripción automática con [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- 🌍 Traducción en tiempo real vía Google Translate (o Argos Translate
  para uso 100% offline).
- 🌐 Auto-detección de idioma de origen.
- 🖥️ Ventana de subtítulos flotante, semi-transparente y arrastrable.
- ⚡ Soporte de aceleración por GPU (NVIDIA CUDA).
- 🎚️ Ajustes para audio muy bajito (susurros, ASMR).

## Instalación

```bash
pip install -r requirements.txt
```

### Aceleración por GPU (opcional, recomendado si tenés GPU NVIDIA)

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Y en el script, configurá:
```python
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
```

### Audio del sistema por sistema operativo

- **Windows**: no necesita configuración extra (usa WASAPI loopback).
- **Linux**: no necesita configuración extra (usa el monitor de PulseAudio/PipeWire).
- **macOS**: Apple no permite captura directa del audio de salida.
  Necesitás instalar [BlackHole](https://existential.audio/blackhole/)
  como dispositivo de audio virtual y configurarlo como `DEVICE_NAME`
  en el script.

## Uso

```bash
python subtitle_translator.py
```

Ajustá la configuración al principio del archivo `subtitle_translator.py`:
idioma de origen/destino, tamaño del modelo Whisper, sensibilidad de
detección de voz, ganancia de audio, etc.

## Licencia

Uso libre y personal.
