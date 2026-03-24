"""Minimal client for interacting with the OpenAI API-compatible text generation server."""

import asyncio
import json
import time
from pathlib import Path
from rich import print

import aiohttp
import modal

MINUTES = 60

def main(prompt: str, twice: bool = True):
    # Look up deployed server to get the URL
    SGLang = modal.Cls.from_name("bootstrap-text-generation", "SGLang")
    url = SGLang().serve.get_web_url()

    print("\n\n")
    print("--------------------------------------------------------------------------------")
    print(f"Running {Path(__file__).name} to invoke the deployed function")
    print("--------------------------------------------------------------------------------")
    print(f"[green]Sending request to {url}[/green]")
    print("[green]Loading Qwen3-4B-Instruct on a cloud GPU and running inference.[/green]")
    print(f"[green]View progress in Modal dashboard: [magenta]{SGLang().serve.get_dashboard_url()}[/magenta][green]")
    print(f"[green]Prompt: {prompt}[/green]")
    messages = [{"role": "user", "content": prompt}]

    start = time.perf_counter()
    response = asyncio.run(send_request(url, messages))
    elapsed = time.perf_counter() - start

    print(f"[green]\n✓ Final token received in {elapsed:.2f} seconds[/green]")
    if twice:
        print(f"[green]\nRepeating request to {url}[/green]")
        print(f"[green]Prompt: {prompt}[/green]")
        start = time.perf_counter()
        response = asyncio.run(send_request(url, messages))
        elapsed = time.perf_counter() - start
        print(f"[green]\n✓ Final token received in {elapsed:.2f} seconds[/green]")


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
                chunk = delta.get("content") or ""
                if chunk:
                    print(chunk, end="", flush=True)
                    full_text += chunk
            except json.JSONDecodeError:
                continue

        print()  # newline after stream
        return full_text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Send a chat completion request to the deployed text generation server."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Explain quantum computing in simple terms.",
        help="Text prompt to send to the model.",
    )
    parser.add_argument(
        "--no-twice",
        action="store_false",
        dest="twice",
        help="Pass this flag to submit the prompt only once.",
    )
    args = parser.parse_args()

    main(args.prompt, args.twice)
