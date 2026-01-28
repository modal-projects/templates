# Simple LLM Inference

An instruction-following language model service running on Modal using SGLang and Qwen3-4B.

## Model

This template uses [Qwen3/Qwen3-4B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8), a fast but knowledgeable language model. Runs on an A100 GPU.

## Deploy

```bash
modal deploy llm-inference/app.py
```

## Usage

### Client script

```bash
# Generate from a prompt
python llm-inference/client.py "Eli5 multi-party computation and how it relates to machine learning! Include a summary of the most important libraries that support both."
```

### Programmatic access

```python
import modal
import openai

llm = modal.Cls.from_name("qwen3-4b-inference", "SGLang")()
modal_endpoint = llm.serve.get_web_url()
base_url = modal_endpoint + "/v1"

client = openai.OpenAI(base_url=base_url, api_key="empty")

response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": "Eli5 multi-party computation and how it relates to machine learning!",
        }
    ],
)

print(response.choices[0].message.content)
```
