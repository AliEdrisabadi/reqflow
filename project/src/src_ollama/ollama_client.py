import json
import time
import os
import requests
from typing import Any, Dict, Optional

from dotenv_loader import load_dotenv

# Load config from .env in the current working directory (if present)
load_dotenv()

class OllamaError(RuntimeError):
    pass

def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    host: Optional[str] = None,
    temperature: Optional[float] = None,
    num_ctx: Optional[int] = None,
    json_mode: Optional[bool] = None,
    timeout: Optional[int] = None,
    retries: Optional[int] = None,
) -> Dict[str, Any]:
    """Call Ollama /api/generate.
    """
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")

    temperature = temperature if temperature is not None else _env_float("OLLAMA_TEMPERATURE", 0.2)
    num_ctx = num_ctx if num_ctx is not None else _env_int("OLLAMA_NUM_CTX", 4096)
    json_mode = json_mode if json_mode is not None else _env_bool("OLLAMA_JSON_MODE", True)
    timeout = timeout if timeout is not None else _env_int("OLLAMA_TIMEOUT", 120)
    retries = retries if retries is not None else _env_int("OLLAMA_RETRIES", 2)

    url = host.rstrip("/") + "/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    if json_mode:
        payload["format"] = "json"

    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            text = data.get("response", "")
            if json_mode:
                return json.loads(text)
            # fallback: try parse anyway
            try:
                return json.loads(text)
            except Exception:
                return {"response": text}
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise OllamaError(f"Ollama call failed: {last_err}")
