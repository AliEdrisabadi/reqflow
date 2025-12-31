from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ollama import ollama_generate


TAGS = [
    "Main_actor",
    "Entity",
    "Action",
    "System_response",
    "Condition",
    "Precondition",
    "Constraint",
    "Exception",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def fill(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def _find_occurrences(text: str, needle: str) -> List[Tuple[int, int]]:
    if not needle:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), text)]


def _normalize_variants(s: str) -> List[str]:
    variants = [s]
    variants.append(s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'"))
    variants.append(s.replace("\u00a0", " ").replace("\xa0", " "))
    out = []
    for v in variants:
        if v and v not in out:
            out.append(v)
    return out


def _repair_span(text: str, span: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cand = str(span.get("text", "") or "")
    if not cand:
        return None

    occs: List[Tuple[int, int]] = []
    for v in _normalize_variants(cand):
        occs.extend(_find_occurrences(text, v))

    if not occs:
        return None

    target = span.get("start", None)
    if isinstance(target, int):
        occs.sort(key=lambda t: abs(t[0] - target))
    else:
        occs.sort(key=lambda t: t[0])

    st, en = occs[0]
    span["start"], span["end"] = st, en
    span["text"] = text[st:en]
    return span


def validate_spans(text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spans = spans or []
    fixed: List[Dict[str, Any]] = []
    seen = set()

    for s in spans:
        if not isinstance(s, dict):
            continue
        tag = s.get("tag")
        if tag not in TAGS:
            continue

        st, en = s.get("start"), s.get("end")
        ok = (
            isinstance(st, int)
            and isinstance(en, int)
            and 0 <= st <= en <= len(text)
            and text[st:en] == (s.get("text") or "")
        )

        if not ok:
            s2 = _repair_span(text, dict(s))
            if s2 is None:
                continue
            s = s2

        key = (s["tag"], int(s["start"]), int(s["end"]), s.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        fixed.append({"tag": s["tag"], "start": int(s["start"]), "end": int(s["end"]), "text": s.get("text", "")})

    fixed.sort(key=lambda x: (x["start"], x["end"], x["tag"]))
    return fixed


def _pick_text_column(df: pd.DataFrame) -> str:
    for c in ["text_en", "text", "requirement", "req"]:
        if c in df.columns:
            return c
    raise ValueError("Dataset CSV must contain a requirement text column (expected: text_en).")


def main(
    dataset_csv: str,
    out_json: str,
    model: Optional[str],
    prompt_path: Optional[str] = None,
    ids: str = "",
) -> None:
    df = pd.read_csv(dataset_csv)
    text_col = _pick_text_column(df)

    if ids.strip():
        wanted = {int(x) for x in ids.split(",") if x.strip().isdigit()}
        df = df[df["id"].isin(wanted)].copy()

    root = _project_root()
    prompt_file = Path(prompt_path).resolve() if prompt_path else (root / "prompts" / "baseline.md").resolve()
    tmpl = load_prompt(prompt_file)

    outputs: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rid = int(row["id"])
        text = str(row[text_col])

        prompt = fill(tmpl, REQUIREMENT_TEXT=text)
        try:
            out = ollama_generate(prompt, model=model)
            spans = out.get("spans", []) if isinstance(out, dict) else []
            spans = validate_spans(text, spans)
            outputs.append({"id": rid, "text": text, "spans": spans})
        except Exception as e:
            outputs.append({"id": rid, "text": text, "spans": [], "error": str(e)})

    Path(out_json).write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} items to {out_json}")
