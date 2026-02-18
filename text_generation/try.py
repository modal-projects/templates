"""Client for interacting with the SGLang inference server."""

import asyncio
import json
import time

import aiohttp

MINUTES = 60


async def send_request(url: str, messages: list, timeout: int = 5 * MINUTES):
    """Send a chat completion request to the server."""
    deadline = time.time() + timeout
    async with aiohttp.ClientSession(base_url=url) as session:
        while time.time() < deadline:
            try:
                return await _send_streaming(session, messages)
            except asyncio.TimeoutError:
                print("Request timed out, retrying...")
                await asyncio.sleep(1)
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

    # Look up deployed server to get the URL
    SGLang = modal.Cls.from_name("bootstrap-text-generation", "SGLang")
    url = SGLang().serve.get_web_url()

    print(f"Sending request to {url}")
    print(f"Prompt: {args.prompt}")
    messages = [{"role": "user", "content": args.prompt}]

    start = time.perf_counter()
    response = asyncio.run(send_request(url, messages))
    elapsed = time.perf_counter() - start

    print(response)
    print(f"\n✓ Response received in {elapsed:.2f} seconds")
