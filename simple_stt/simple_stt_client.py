"""
Client for the SimpleSTT Modal service.

Usage:
    python simple_stt/simple_stt_client.py [url]

Examples:
    python simple_stt/simple_stt_client.py
    python simple_stt/simple_stt_client.py https://example.com/audio.wav
"""

import argparse
import json
import time
from urllib.request import urlopen
from urllib.parse import urlencode

import modal

DEFAULT_AUDIO_URL = "https://modal-cdn.com/a_dream_within_a_dream_16000_mono.wav"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using SimpleSTT.")
    parser.add_argument(
        "url",
        nargs="?",
        default=DEFAULT_AUDIO_URL,
        help="URL to a WAV file (16kHz, mono). Uses test audio if omitted.",
    )
    args = parser.parse_args()

    # Call via Modal SDK
    print("=== Modal SDK ===")
    stt = modal.Cls.from_name("simple-stt-template", "SimpleSTT")()

    start = time.perf_counter()
    transcript = stt.transcribe.remote(args.url)
    elapsed = time.perf_counter() - start

    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Transcript: {transcript}")

    # Call via HTTP endpoint
    print("\n=== HTTP Endpoint ===")
    endpoint_url = stt.api.get_web_url()

    start = time.perf_counter()
    url_with_params = f"{endpoint_url}?{urlencode({'audio': args.url})}"
    with urlopen(url_with_params) as response:
        transcript = json.load(response)
    elapsed = time.perf_counter() - start

    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Transcript: {transcript}")
