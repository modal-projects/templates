# Simple Speech-to-Text

A speech-to-text service running on Modal using NVIDIA's Parakeet ASR model.

## Model

This template uses [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), a fast and accurate multilanguage ASR model.

## Deploy

```bash
modal deploy -m simple-stt.simple_stt_app
```

## Input formats

The `transcribe` method accepts:
- **URL** — a URL pointing to a WAV file (16kHz, mono)
- **Raw bytes** — 16-bit PCM at 16kHz, mono

## Client

### Modal SDK 

```python
import modal

stt = modal.Cls.from_name("simple-stt-template", "SimpleSTT")()

# From a URL (WAV file, 16kHz, mono)
transcript = stt.transcribe.remote("https://example.com/audio.wav")

# From raw PCM bytes (16-bit, 16kHz, mono)
transcript = stt.transcribe.remote(audio_bytes)
```

### FastAPI HTTP endpoint

```python
import modal
from urllib.request import urlopen
from urllib.parse import urlencode
import json

stt = modal.Cls.from_name("simple-stt-template", "SimpleSTT")()
endpoint_url = stt.api.get_web_url()

url_with_params = f"{endpoint_url}?{urlencode({'audio': 'https://example.com/audio.wav'})}"
with urlopen(url_with_params) as response:
    transcript = json.load(response)
```

### Example
Both approaches are demonstrated in `simple_stt_client.py`.

To run the example client:

```bash
# Transcribe a URL (uses test audio if omitted)
python simple-stt/simple_stt_client.py https://example.com/audio.wav
```


