# Simple Speech-to-Text

A speech-to-text service running on Modal using NVIDIA's Parakeet ASR model.

## Model

This template uses [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), a fast and accurate multilanguage ASR model. Runs on an L40S. Inference takes ~45 seconds to [cold start](https://modal.com/docs/guide/cold-start).

## Deploy

```bash
modal deploy -m speech_to_text.app
```

## Input formats

The `transcribe` method accepts:

- **URL** — a URL pointing to a WAV file (16kHz, mono)
- **Raw bytes** — 16-bit PCM at 16kHz, mono

## Usage

### Programmatic access

```bash
# Sample client script to generate a transcript (uses test audio if omitted)
python speech_to_text/try.py "https://modal-cdn.com/a_dream_within_a_dream_16000_mono.wav"
```

### Web

A [web endpoint](https://modal.com/docs/guide/webhook-urls) is automatically generated for this inference function. The URL is printed out when you deploy the app; you can also find it in the Modal web dashboard under the page for the function.

```bash
curl \
  --get '<URL>' \
  --data-urlencode 'audio=<AUDIO FILE URL>'
```
