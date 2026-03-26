# LLM Providers

POIROT supports 4 LLM providers. Select one via the `provider` parameter in `run_poirot()`.

---

## Gemini (Google)

```python
results = poirot.run_poirot(
    ...
    provider="gemini",
    api_key="YOUR_GOOGLE_API_KEY",
)
```

- **Default model:** `gemini-2.5-pro`
- **API key env var:** `GOOGLE_API_KEY`
- Get your key at [aistudio.google.com](https://aistudio.google.com)

---

## DeepSeek

```python
results = poirot.run_poirot(
    ...
    provider="deepseek",
    api_key="YOUR_DEEPSEEK_API_KEY",
)
```

- **Default model:** `deepseek-chat`
- **API key env var:** `DEEPSEEK_API_KEY`

---

## Ollama (local)

Requires [Ollama](https://ollama.com) running locally on port `11434`. No API key needed.

```python
results = poirot.run_poirot(
    ...
    provider="ollama",
    model="llama3.1:8b",       # any model you have pulled
    ollama_num_ctx=131072,     # context window in tokens
)
```

- **Default model:** `gpt-oss:20b`
- **Endpoint:** `http://localhost:11434/v1`
- **No API key required**

Pull a model before running:
```bash
ollama pull llama3.1:8b
```

---

## Local / LM Studio

Requires [LM Studio](https://lmstudio.ai) running locally on port `1234` with the local server enabled. No API key needed.

```python
results = poirot.run_poirot(
    ...
    provider="local",
    model="your-loaded-model-name",
)
```

- **Default model:** `gpt-oss-20b`
- **Endpoint:** `http://localhost:1234/v1`
- **No API key required**

---

## Default models summary

| Provider | Default model |
|----------|--------------|
| `gemini` | `gemini-2.5-pro` |
| `deepseek` | `deepseek-chat` |
| `ollama` | `gpt-oss:20b` |
| `local` | `gpt-oss-20b` |

Override any default with the `model` parameter.
