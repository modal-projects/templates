"""
Client for the SimpleSTT Modal service.

Usage:
    python simple_stt/simple_stt_client.py [url]

Examples:
    python simple_stt/simple_stt_client.py # uses default test audio
    python simple_stt/simple_stt_client.py https://example.com/audio.wav
"""

import argparse
import time

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

    # Connect to the deployed SimpleSTT service
    stt = modal.Cls.from_name("simple-stt-template", "SimpleSTT")()

    start = time.perf_counter()
    transcript = stt.transcribe.remote(args.url)
    elapsed = time.perf_counter() - start

    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Transcript: {transcript}")
