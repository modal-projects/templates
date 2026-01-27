"""
Speech-to-text service using NVIDIA Parakeet on Modal.

Deploy: modal deploy simple-stt/simple_stt_app.py

Usage:
    stt = modal.Cls.from_name("simple-stt-template", "SimpleSTT")()
    transcript = stt.transcribe.remote("https://example.com/audio.wav")
"""

import logging
import time

import modal

app = modal.App(name="simple-stt-template")

cache_volume = modal.Volume.from_name("stt-template-cache", create_if_missing=True)
CACHE_DIR = "/cache"

MINUTES = 60  # seconds

TEST_AUDIO_URL = "https://modal-cdn.com/a_dream_within_a_dream_16000_mono.wav"

# Audio format requirements: 16kHz sample rate, mono, 16-bit PCM
SAMPLE_RATE = 16000

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_DIR,  # cache directory for Hugging Face models
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg")
    .uv_pip_install(
        "huggingface_hub[hf-xet]==0.31.2",
        "nemo_toolkit[asr]==2.3.2",
        "cuda-python==13.0.1",
        "soundfile==0.13.1",
        "requests==2.32.5",
    )
    .entrypoint([])  # silence chatty logs by container on start
)

with image.imports():
    import nemo.collections.asr as nemo_asr
    import torch
    import soundfile as sf
    import numpy as np
    import requests
    import io

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


@app.cls(
    gpu="L40S",
    image=image,
    volumes={
        CACHE_DIR: cache_volume,
    },
)
class SimpleSTT:
    """Transcribes audio files or URLs using NVIDIA's Parakeet ASR model."""

    @modal.enter()
    def setup(self):
        """Load the ASR model and warm up the GPU on container start."""
        self._dtype = torch.bfloat16

        # silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        self.asr_model.to(self._dtype)
        self.asr_model.eval()

        # Configure decoding strategy
        if self.asr_model.cfg.decoding.strategy != "beam":
            self.asr_model.cfg.decoding.strategy = "greedy_batch"
            self.asr_model.change_decoding_strategy(self.asr_model.cfg.decoding)

        # run test request to warm up GPU
        for _ in range(4):
            self.transcribe.local(TEST_AUDIO_URL)

    @modal.method()
    def transcribe(self, audio: bytes | str) -> str | list[str]:
        """
        Converts speech audio to text using NVIDIA's Parakeet ASR model.

        This method accepts either a URL pointing to a 16kHz, mono WAV file or raw PCM audio bytes
        (16-bit signed int, 16kHz, mono). It automatically loads and decodes the audio, processes
        it with the ASR model, and returns the textual transcription(s).

        Args:
            audio: Either a direct URL to a 16kHz, mono WAV file, or raw PCM audio bytes
                   (16-bit signed int, 16kHz, mono).

        Returns:
            str: The transcription if a single utterance is detected.
            list[str]: A list of transcriptions if multiple utterances are detected.
        """

        t0 = time.time()

        if isinstance(audio, str):
            # Fetch WAV from URL and decode
            audio_content = requests.get(audio).content
            audio_obj = io.BytesIO(audio_content)
            try:
                audio, sample_rate = sf.read(audio_obj, dtype="float32")
                if sample_rate != SAMPLE_RATE:
                    raise ValueError(
                        f"Sample rate mismatch: {sample_rate} != {SAMPLE_RATE}"
                    )
            except Exception as e:
                raise ValueError(
                    f"Error reading audio file from URL ({audio}): {type(e).__name__}: {e}"
                )
        else:
            # Convert raw PCM bytes to float32 normalized to [-1, 1]
            audio = np.frombuffer(audio, dtype=np.int16)
            audio = audio.astype(np.float32) / 32767.0

        with (
            torch.autocast("cuda", enabled=False, dtype=self._dtype),
            torch.inference_mode(),
            torch.no_grad(),
        ):
            results = self.asr_model.transcribe(audio)

        transcripts = [result.text for result in results]

        t1 = time.time()
        print(f"Transcription time for input: {t1 - t0} seconds")
        if len(transcripts) == 1:
            return transcripts[0]
        else:
            return transcripts
