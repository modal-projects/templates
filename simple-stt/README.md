# Simple Speech-to-Text

A speech-to-text service running on Modal using NVIDIA's Parakeet ASR model.

## Model

This template uses [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), a fast and accurate English ASR model. Runs on an L40S GPU.

## Deploy

```bash
modal deploy -m simple-stt.simple_stt_app
```

## Usage

### Client script

```bash
# Transcribe a URL (uses test audio if omitted)
python simple-stt/simple_stt_client.py https://example.com/audio.mp3

# Transcribe a local file
python simple-stt/simple_stt_client.py recording.wav
```

### Programmatic access

```python
import modal

stt = modal.Cls.from_name("simple-stt-template", "SimpleSTT")()

# From a URL
transcript = stt.transcribe.remote("https://example.com/audio.mp3")

# From a local file path
transcript = stt.transcribe.remote("/path/to/audio.wav")

# From raw PCM bytes (16-bit, 16kHz, mono)
transcript = stt.transcribe.remote(audio_bytes)
```

## Input formats

The `transcribe` method accepts:
- **URL** — any audio format (converted via ffmpeg)
- **File path** — any audio format (converted via ffmpeg)
- **Raw bytes** — 16-bit PCM at 16kHz, mono
