from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence


def find_dotenv(
    start: str | os.PathLike | None = None,
    filename: str = ".env",
    *,
    fallback_filenames: Sequence[str] = ("reqflow.env", ".env.example"),
) -> Optional[str]:
    """
    Find an env file by walking upward from `start`.

    We primarily look for `.env`, but some environments (or repos) block dotfiles.
    In that case, we also accept `reqflow.env` (recommended) or `.env.example`.
    """
    cur = Path(start or Path.cwd()).resolve()
    candidates = (filename, *tuple(fallback_filenames))
    for p in [cur] + list(cur.parents):
        for name in candidates:
            cand = p / name
            if cand.exists() and cand.is_file():
                return str(cand)
    return None


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def load_dotenv(path: str | os.PathLike | None = None, override: bool = False) -> None:
    if path is None:
        path = find_dotenv()
        if path is None:
            return

    p = Path(path).expanduser().resolve()
    if not p.exists():
        return

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        key = k.strip()
        val = _strip_quotes(v.strip())

        if not key:
            continue
        if (not override) and (key in os.environ):
            continue

        os.environ[key] = val
