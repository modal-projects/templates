"""Client for interacting with the SGLang inference server."""

import asyncio
import json
import time

import aiohttp

MINUTES = 60


async def send_request(
    url: str, messages: list, timeout: int = 5 * MINUTES, stream: bool = True
):
    """Send a chat completion request to the server."""
    deadline = time.time() + timeout
    async with aiohttp.ClientSession(base_url=url) as session:
        while time.time() < deadline:
            try:
                if stream:
                    return await _send_streaming(session, messages)
                else:
                    return await _send_non_streaming(session, messages)
            except asyncio.TimeoutError:
                print("Request timed out, retrying...")
                await asyncio.sleep(1)
            except aiohttp.ClientResponseError as e:
                if e.status == 503:
                    print("Server starting up (503), retrying...")
                    await asyncio.sleep(1)
                    continue
                raise
    raise TimeoutError(f"No response within {timeout}s")


async def _send_streaming(session: aiohttp.ClientSession, messages: list):
    """Send streaming chat completion request and print chunks."""
    payload = {"messages": messages, "stream": True}
    headers = {"Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers
    ) as resp:
        resp.raise_for_status()
        full_text = ""

        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break

            try:
                evt = json.loads(data)
                delta = (evt.get("choices") or [{}])[0].get("delta", {})
                chunk = delta.get("content")
                if chunk:
                    print(chunk, end="", flush=True)
                    full_text += chunk
            except json.JSONDecodeError:
                continue

        print()  # newline after stream
        return full_text


async def _send_non_streaming(session: aiohttp.ClientSession, messages: list):
    """Send non-streaming chat completion request."""
    payload = {"messages": messages, "stream": False}

    async with session.post("/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        result = await resp.json()
        content = result["choices"][0]["message"]["content"]
        print(content)
        return content


if __name__ == "__main__":
    import argparse

    import modal

    parser = argparse.ArgumentParser(
        description="Send a chat completion request to the deployed SGLang server."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Explain quantum computing in simple terms.",
        help="Text prompt to send to the model.",
    )
    args = parser.parse_args()

    # Look up deployed server
    Deployed = modal.Cls.from_name("qwen3-4b-inference", "SGLang")
    url = Deployed().serve.get_web_url()

    print(f"Sending request to {url}")
    messages = [{"role": "user", "content": args.prompt}]

    start = time.perf_counter()
    response = asyncio.run(send_request(url, messages))
    elapsed = time.perf_counter() - start

    print(f"\n✓ Response received in {elapsed:.2f} seconds")
    print("from Qwen3-4B running on Modal with SGLang")
    print(f"Prompt: {args.prompt}")
