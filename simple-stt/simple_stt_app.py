"""
Speech-to-text service using NVIDIA Parakeet on Modal.

Deploy with: modal deploy simple_stt_app.py
"""
import os
import logging
import subprocess

import modal

app = modal.App(name="simple-stt-template")

cache_volume = modal.Volume.from_name("stt-template-cache", create_if_missing=True)
CACHE_DIR = "/cache"

MINUTES = 60 # seconds

TEST_AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_DIR,  # cache directory for Hugging Face models
            "DEBIAN_FRONTEND": "noninteractive",
            "CXX": "g++",
            "CC": "g++",
            "TORCH_HOME": CACHE_DIR,
        }
    )
    .apt_install("ffmpeg")
    .uv_pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==13.0.1",
        "numpy<2",
        "torchaudio",
        "soundfile",
        "resampy",
        "fastapi[standard]",
    )
    .entrypoint([])  # silence chatty logs by container on start
)


with image.imports():
    import nemo.collections.asr as nemo_asr
    import torch
    import tempfile
    import soundfile as sf
    import numpy as np
    
    from urllib.request import urlopen


MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"

@app.cls(
    gpu="L40S",
    image=image, 
    volumes={
        CACHE_DIR: cache_volume,
    },
)
class SimpleSTT():
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

        self.warm_up_gpu()

    async def warm_up_gpu(self):
        """Run a test transcription to warm up CUDA kernels/cache."""

        print("Warming up GPU...")
        
        audio_bytes = self.preprocess_audio(TEST_AUDIO_URL, target_sample_rate=SAMPLE_RATE)
        
        # Then chunk the audio data (not the raw bytes)
        chunk_size_seconds = 10
        chunk_size = SAMPLE_RATE * chunk_size_seconds * SAMPLE_WIDTH_BYTES  # at 16kHz
        audio_chunks = [audio_bytes[i:i+chunk_size] for i in range(0, len(audio_bytes), chunk_size)]

        # batch the chunks and perform transcription
        for chunk in audio_chunks:
            await self.transcribe.local(chunk)

    @modal.method()
    async def transcribe(self, audio: bytes | str) -> str | list[str]:
        """
        Transcribe audio to text.
        
        Args:
            audio: URL, file path, or raw 16-bit PCM bytes at 16kHz.
            
        Returns:
            Transcript string, or list of strings if multiple chunks provided.
        """
        if isinstance(audio, str):
            audio = self.preprocess_audio(audio)
        else:
            audio = np.frombuffer(audio, dtype=np.int16)
            audio = audio.astype(np.float32) / 32767.0
            
        with torch.autocast("cuda", enabled=False, dtype=self._dtype), torch.inference_mode(), torch.no_grad():
            results = self.asr_model.transcribe(audio)

        transcripts = [result.text for result in results]
        if len(transcripts) == 1:
            return transcripts[0]
        else:
            return transcripts

    def preprocess_audio(self, audio_src: str, target_sample_rate: int = 16000):
        """
        Load and convert audio to float32 waveform via ffmpeg.
        
        Args:
            audio_src: URL or local file path.
            target_sample_rate: Output sample rate (default 16kHz).
            
        Returns:
            Float32 numpy array normalized to [-1, 1].
        """
        if audio_src.startswith("http"):
            audio_src = urlopen(audio_src).read()
    
            with tempfile.NamedTemporaryFile(suffix='.input', delete=False) as tmp_in:
                tmp_in.write(audio_src)
                tmp_in_path = tmp_in.name
        else:
            tmp_in_path = audio_src

                
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
        
        try:
            # Use ffmpeg to convert to WAV
            subprocess.run(
                [
                    'ffmpeg', '-i', tmp_in_path,
                    '-ar', str(target_sample_rate),  # Set sample rate
                    '-ac', '1',  # Mono
                    '-f', 'wav',  # Output format
                    '-y',  # Overwrite
                    tmp_out_path
                ],
                capture_output=True,
                check=True
            )
            
            # Load the converted WAV
            waveform, sample_rate = sf.read(tmp_out_path, dtype='float32')
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to load audio with ffmpeg: {e.stderr.decode()}")
        finally:
            os.unlink(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.unlink(tmp_out_path)

        return waveform
