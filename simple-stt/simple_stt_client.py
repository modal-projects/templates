"""
Client for the SimpleSTT Modal service.

Usage:
    python simple_stt_client.py [audio_source]

Examples:
    python simple_stt_client.py                        # uses default test audio
    python simple_stt_client.py recording.wav          # local file
    python simple_stt_client.py https://example.com/a.mp3
"""
import argparse
import time

import modal

if __name__ == "__main__":
    """
    Transcribe an audio file or URL using the deployed SimpleSTT service.
    
    Args:
        audio_source: Path to a local audio file or URL (optional, uses test audio if omitted).
    """

    # Default test audio URL from the app
    DEFAULT_AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"

    parser = argparse.ArgumentParser(
        description="Transcribe audio using SimpleSTT."
    )
    parser.add_argument(
        "audio_source",
        nargs="?",
        default=DEFAULT_AUDIO_URL,
        help="Path to an audio file or URL to transcribe.",
    )
    args = parser.parse_args()

    audio_source = args.audio_source

    stt = modal.Cls.from_name("simple-stt-template", "SimpleSTT")()

    start = time.perf_counter()
    transcript = stt.transcribe.remote(audio_source)
    elapsed = time.perf_counter() - start

    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Transcript: {transcript}")
