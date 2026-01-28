import subprocess
import time

import modal

MINUTES = 60

# Model configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507-FP8"
MODEL_REVISION = "8591804019c8b22094c3b5b4454e0edc05dffc98"
GPU = "A100"
PORT = 8000

# Volumes for caching
HF_CACHE_PATH = "/root/.cache/huggingface"
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


# Image setup
sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.8-cu129-amd64-runtime")
    .entrypoint([])
    .uv_pip_install("huggingface-hub==0.36.0")
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)


app = modal.App("qwen3-4b-inference")


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
)
@modal.concurrent(max_inputs=32)
class SGLang:
    @modal.enter()
    def startup(self):
        """Start SGLang server, wait for health, and warm up."""
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
        ]
        self.process = subprocess.Popen(cmd)
        self._wait_ready()
        self._warmup()

    def _wait_ready(self, timeout=5 * MINUTES):
        """Block until SGLang server is healthy."""
        import requests

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited with code {self.process.returncode}"
                    )
                requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
                return
            except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
                time.sleep(5)
        raise TimeoutError(f"Server not ready within {timeout}s")

    def _warmup(self):
        """Send warmup requests to populate caches."""
        import requests

        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        }
        for _ in range(3):
            requests.post(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json=payload,
                timeout=30,
            ).raise_for_status()

    @modal.exit()
    def shutdown(self):
        """Terminate the SGLang server process."""
        self.process.terminate()

    @modal.web_server(port=PORT, startup_timeout=5 * MINUTES)
    def serve(self):
        "Stub method to denote the `web_server` that was initialized during startup()"
        pass


@app.local_entrypoint()
async def test(prompt: str = None, timeout: int = 10):
    """Test the deployed server."""
    from llm_client import send_request

    # Try to look up deployed app first, fall back to ephemeral
    try:
        Deployed = modal.Cls.from_name("qwen3-4b-inference", "SGLang")
        url = Deployed().serve.get_web_url()
        print(f"Using deployed server at {url}")
    except modal.exception.NotFoundError:
        url = SGLang().serve.get_web_url()
        print(f"Using ephemeral server at {url}")

    if prompt is None:
        prompt = "Explain quantum computing in simple terms."

    messages = [{"role": "user", "content": prompt}]
    await send_request(url, messages, timeout=timeout * MINUTES)
