from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from huggingface_hub import InferenceClient
try:
    # huggingface_hub>=1.x
    from huggingface_hub import get_token as _hf_get_token
except Exception:  # pragma: no cover
    _hf_get_token = None

try:
    # huggingface_hub<1.x
    from huggingface_hub import HfFolder as _HfFolder
except Exception:  # pragma: no cover
    _HfFolder = None


@dataclass
class OllamaConfig:
    host: str
    model: str
    timeout: float
    temperature: float
    num_predict: int


class OllamaError(RuntimeError):
    pass


def _is_hf_model_id(model: str) -> bool:
    m = (model or "").strip()
    # Treat a HF repo id (org/name) as a signal to use HF inference.
    return bool(m) and ("/" in m) and (not m.startswith("http"))


def _normalize_ollama_model(model: str) -> str:
    """
    ReqFlow historically uses Ollama model names (e.g. "llama3.1:8b").

    Users sometimes provide Hugging Face model IDs (e.g.
    "meta-llama/Llama-3.1-8B-Instruct"). For convenience, map a few common
    HF IDs to the closest Ollama model name.

    Note: This does NOT download HF models. It only changes the string sent
    to an already-running Ollama server.
    """
    m = (model or "").strip()
    if not m:
        return m

    key = m.lower()
    # Common HF repos for Llama 3.1 8B instruct/chat.
    if key in {
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/meta-llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-8b",
        "meta-llama/meta-llama-3.1-8b",
    }:
        return "llama3.1:8b"

    return m


def _env_token(*keys: str) -> Optional[str]:
    for k in keys:
        v = os.getenv(k)
        if v and v.strip():
            return v.strip()

    # Fall back to the Hugging Face cached token created by `hf auth login`.
    try:
        if _hf_get_token is not None:
            t = _hf_get_token()
        elif _HfFolder is not None:
            t = _HfFolder.get_token()
        else:
            t = None
        if t and str(t).strip():
            return str(t).strip()
    except Exception:
        pass
    return None


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
        # Accept HF-style ID; we normalize to an Ollama name right before calling.
        model=_env_str("OLLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        timeout=_env_float("OLLAMA_TIMEOUT", 120.0),
        temperature=_env_float("OLLAMA_TEMPERATURE", 0.2),
        num_predict=_env_int("OLLAMA_NUM_PREDICT", 512),
    )


def ollama_check(timeout: float = 5.0) -> Tuple[bool, str]:
    cfg = get_config()

    # If the configured model looks like a Hugging Face repo id, don't block the app
    # on a local Ollama daemon. We'll attempt HF generation at request time.
    if _is_hf_model_id(cfg.model):
        tok = _env_token("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_TOKEN")
        if tok:
            return True, f"Hugging Face Inference (model: {cfg.model})"
        return True, f"Hugging Face Inference (model: {cfg.model}, token: missing)"

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


def _hf_generate(
    prompt: str,
    model: str,
    *,
    temperature: float,
    max_new_tokens: int,
    timeout: float,
    retries: int,
) -> Dict[str, Any]:
    tok = _env_token("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_TOKEN")
    if not tok:
        raise OllamaError(
            "HF model ID provided but no token found. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN). "
            "Note: meta-llama models are gated and require access on Hugging Face."
        )

    # Use Hugging Face hosted inference (no local weights download).
    client = InferenceClient(token=tok, timeout=timeout)

    def _wrap_non_dict(obj: Any) -> Dict[str, Any]:
        # Some providers/models occasionally return a bare JSON array.
        # Wrap common shapes so downstream code can proceed.
        if isinstance(obj, list):
            if all(isinstance(x, dict) for x in obj):
                if any("tag" in x for x in obj):
                    return {"spans": obj}
                if any(("clause_id" in x) or ("cue" in x) for x in obj):
                    return {"clauses": obj}
            return {"items": obj}
        return {"value": obj}

    def _chat_completion_text() -> str:
        # Prefer forcing a JSON object response, but some providers don't support it.
        try:
            out = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                stream=False,
                temperature=temperature,
                max_tokens=max_new_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            msg = str(e).lower()
            if "response_format" in msg or "json" in msg:
                out = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
            else:
                raise

        # huggingface_hub returns a typed object that supports attribute access.
        try:
            return str(out.choices[0].message.content)
        except Exception:
            return str(out)

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            try:
                txt = client.text_generation(
                    prompt,
                    model=model,
                    stream=False,
                    details=False,
                    return_full_text=False,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature is not None and float(temperature) > 0.0),
                )
            except Exception as e:
                # Some providers only expose this model on the "conversational" (chat) task.
                msg = str(e).lower()
                if "supported task" in msg and "conversational" in msg:
                    txt = _chat_completion_text()
                else:
                    raise
            obj = _extract_json(txt if isinstance(txt, str) else str(txt))
            return obj if isinstance(obj, dict) else _wrap_non_dict(obj)
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(0.6 * (attempt + 1))

    raise OllamaError(f"HF generate failed after {retries + 1} attempts: {last_err}")


def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    retries: int = 2,
) -> Dict[str, Any]:
    cfg = get_config()
    raw_model = (model or cfg.model).strip()

    # If user supplied a Hugging Face model id (org/name), run via HF inference.
    if _is_hf_model_id(raw_model):
        return _hf_generate(
            prompt,
            raw_model,
            temperature=cfg.temperature if temperature is None else float(temperature),
            max_new_tokens=cfg.num_predict if num_predict is None else int(num_predict),
            timeout=cfg.timeout,
            retries=retries,
        )

    m = _normalize_ollama_model(raw_model)
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
