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

    stt = modal.Cls.from_name("bootstrap-text-to-speech", "STT")()

    print("Loading NVIDIA Parakeet on a cloud L40S GPU.")
    print(f"View progress in Modal dashboard: {stt.transcribe.get_dashboard_url()}.")

    start = time.perf_counter()
    transcript = stt.transcribe.remote(args.url)
    elapsed = time.perf_counter() - start

    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Transcript: {transcript}")

    print("\nTranscribe another audio file by running: python text_to_speech/try.py \"<AUDIO FILE URL>\"")