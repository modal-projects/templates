import argparse
import json
import time
from urllib.request import urlopen
from urllib.parse import urlencode
from pathlib import Path
from rich import print
from rich.console import Console

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

    # print(f"\n\n===== Running {Path(__file__).name} to invoke the deployed function =====")
    print("\n\n")
    print("--------------------------------------------------------------------------------")
    print(f"Running {Path(__file__).name} to invoke the deployed function")
    print("--------------------------------------------------------------------------------")

    start = time.perf_counter()
    with Console().status(
        (
            "Loading NVIDIA Parakeet on a cloud GPU and running inference.\n"
            f"View progress in Modal dashboard: [magenta]{stt.transcribe.get_dashboard_url()}[/magenta]"
        ),
        spinner="dots",
    ):
        transcript = stt.transcribe.remote(args.url)
    elapsed = time.perf_counter() - start

    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Transcript: {transcript}")

    print(f"\nTranscribe another audio file by running: python text_to_speech/{Path(__file__).name} \"<AUDIO FILE URL>\"")