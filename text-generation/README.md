# Simple Text Generation

An instruction-following language model service running on Modal using SGLang and Qwen3-4B.

## Model

This template uses [Qwen3/Qwen3-4B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8), a fast but knowledgeable language model. Runs on an A100 GPU.

## Deploy

```bash
modal deploy text-generation/app.py
```

## Usage

### Client script

```bash
# Generate from a prompt
python text-generation/client.py "What is multi-party computation? ELI5, ELI20, and ELIPhD."
```

### Programmatic access

The language model service exposes an OpenAI-compatible API,
so you can integrate it with clients that speak that API,
like the `openai` Python SDK, the Vercel AI SDK, and OpenCode.
