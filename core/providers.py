import os
import time
from dataclasses import dataclass

import requests


@dataclass
class LLMResult:
    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tokens_per_second: float


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = "qwen2.5:0.5b"

PRICES = {
    "ollama": {"in": 0.0, "out": 0.0},
}


def count_tokens(text: str) -> int:
    return max(1, int(len(str(text).split()) * 1.3))


def complete(
    prompt: str,
    provider: str = "ollama",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    num_predict: int = 260,
) -> LLMResult:
    provider = (provider or "ollama").lower().strip()
    model = model or DEFAULT_MODEL

    if provider != "ollama":
        raise ValueError(
            f"Unsupported provider '{provider}'. This project is configured for Ollama only."
        )

    start = time.perf_counter()

    text = ollama_generate(
        prompt=prompt,
        model=model,
        temperature=temperature,
        num_predict=num_predict,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(text)
    latency_seconds = max(latency_ms / 1000, 0.001)
    tokens_per_second = round(output_tokens / latency_seconds, 2)

    price = PRICES["ollama"]
    cost_usd = input_tokens * price["in"] + output_tokens * price["out"]

    return LLMResult(
        text=text,
        latency_ms=round(latency_ms, 4),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost_usd, 8),
        tokens_per_second=tokens_per_second,
    )


def ollama_generate(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    num_predict: int = 260,
) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "top_p": 0.9,
                },
            },
            timeout=240,
        )

        response.raise_for_status()
        data = response.json()

        text = data.get("response", "").strip()

        if not text:
            raise RuntimeError(f"Ollama returned an empty response: {data}")

        return text

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is running. Try: ollama serve"
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama timed out while running model '{model}'. Try a smaller model or increase timeout."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        error_text = exc.response.text if exc.response is not None else str(exc)
        raise RuntimeError(f"Ollama HTTP error for model '{model}': {error_text}") from exc