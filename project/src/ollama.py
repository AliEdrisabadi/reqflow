from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


@dataclass
class OllamaConfig:
    host: str
    model: str
    timeout: float
    temperature: float
    num_predict: int


class OllamaError(RuntimeError):
    pass


def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v.strip() if v and v.strip() else default


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def get_config() -> OllamaConfig:
    return OllamaConfig(
        host=_env_str("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        model=_env_str("OLLAMA_MODEL", "llama3.1:8b"),
        timeout=_env_float("OLLAMA_TIMEOUT", 120.0),
        temperature=_env_float("OLLAMA_TEMPERATURE", 0.2),
        num_predict=_env_int("OLLAMA_NUM_PREDICT", 512),
    )


def ollama_check(timeout: float = 5.0) -> Tuple[bool, str]:
    cfg = get_config()
    url = f"{cfg.host}/api/version"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return False, f"Ollama not ready: HTTP {r.status_code} from {url}"
        try:
            v = r.json().get("version", "unknown")
        except Exception:
            v = "unknown"
        return True, f"Ollama OK (version: {v}, model: {cfg.model})"
    except Exception as e:
        return False, f"Cannot reach Ollama at {url}. Is 'ollama serve' running? Error: {e}"


def _extract_json(text: str) -> Any:
    if text is None:
        raise ValueError("Empty response")
    s = text.strip()
    if not s:
        raise ValueError("Empty response")

    if "```" in s:
        parts = s.split("```")
        for i in range(len(parts) - 1):
            header = parts[i].strip().lower()
            body = parts[i + 1]
            if header.endswith("json") or header == "json":
                return json.loads(body.strip())

    start = None
    for ch in "{[":
        idx = s.find(ch)
        if idx != -1:
            start = idx if start is None else min(start, idx)

    if start is None:
        raise ValueError("No JSON found in response")

    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(s[start:])
    return obj


def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    retries: int = 2,
) -> Dict[str, Any]:
    cfg = get_config()
    m = model or cfg.model
    url = f"{cfg.host}/api/generate"

    payload: Dict[str, Any] = {
        "model": m,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": cfg.temperature if temperature is None else temperature,
            "num_predict": cfg.num_predict if num_predict is None else num_predict,
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=cfg.timeout)
            if r.status_code != 200:
                raise OllamaError(f"HTTP {r.status_code} from {url}: {r.text[:500]}")

            txt = r.json().get("response", "")
            obj = _extract_json(txt)

            if not isinstance(obj, dict):
                raise OllamaError("Model output is not a JSON object (dict).")

            return obj

        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(0.6 * (attempt + 1))

    raise OllamaError(f"Ollama generate failed after {retries + 1} attempts: {last_err}")
