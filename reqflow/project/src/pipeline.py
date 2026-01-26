from __future__ import annotations

import os
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


def _resolve_pipeline_prompts(
    segment_prompt_path: Optional[str | Path],
    tag_prompt_path: Optional[str | Path],
) -> Tuple[Path, Path]:
    """
    Resolve pipeline prompt paths.

    Priority for each prompt:
      1) explicit argument
      2) .env: REQFLOW_SEGMENT_PROMPT / REQFLOW_TAG_PROMPT (absolute or relative to REQFLOW_PROMPTS_DIR)
      3) fallback: <root>/<REQFLOW_PROMPTS_DIR or 'prompts'>/segment.md and tag.md
    """
    root = _project_root()
    prompts_dir = root / os.getenv("REQFLOW_PROMPTS_DIR", "prompts")

    def _res(p: Optional[str | Path], env_key: str, fallback: str) -> Path:
        if p:
            pp = Path(p)
            if not pp.is_absolute():
                pp = prompts_dir / pp
            pp = pp.resolve()
            if not pp.exists():
                raise FileNotFoundError(f"Prompt not found: {pp}")
            return pp

        env_p = os.getenv(env_key, "").strip()
        if env_p:
            pp = Path(env_p)
            if not pp.is_absolute():
                pp = prompts_dir / pp
            pp = pp.resolve()
            if not pp.exists():
                raise FileNotFoundError(f"Prompt not found ({env_key}): {pp}")
            return pp

        pp = (prompts_dir / fallback).resolve()
        if not pp.exists():
            raise FileNotFoundError(f"Prompt not found. Set {env_key}. Tried: {pp}")
        return pp

    p1 = _res(segment_prompt_path, "REQFLOW_SEGMENT_PROMPT", "segment.md")
    p2 = _res(tag_prompt_path, "REQFLOW_TAG_PROMPT", "tag.md")
    return p1, p2


def validate_clauses(requirement_text: str, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(clauses, list) or not clauses:
        return [{"clause_id": 1, "start": 0, "end": len(requirement_text), "text": requirement_text, "cue": "MAIN"}]

    good: List[Dict[str, Any]] = []
    for c in clauses:
        if not isinstance(c, dict):
            continue

        cid = c.get("clause_id", c.get("id", len(good) + 1))
        st, en = c.get("start"), c.get("end")
        tx = c.get("text", "")

        if isinstance(st, int) and isinstance(en, int) and 0 <= st <= en <= len(requirement_text):
            if requirement_text[st:en] == tx:
                good.append(
                    {
                        "clause_id": int(cid),
                        "start": st,
                        "end": en,
                        "text": tx,
                        "cue": str(c.get("cue", "")),
                    }
                )
                continue

        cand = str(tx or "").strip()
        if cand:
            m = re.search(re.escape(cand), requirement_text)
            if m:
                st2, en2 = m.start(), m.end()
                good.append(
                    {
                        "clause_id": int(cid),
                        "start": st2,
                        "end": en2,
                        "text": requirement_text[st2:en2],
                        "cue": str(c.get("cue", "")),
                    }
                )

    if not good:
        return [{"clause_id": 1, "start": 0, "end": len(requirement_text), "text": requirement_text, "cue": "MAIN"}]

    good.sort(key=lambda x: (x["start"], x["end"]))
    return good


def _find_occurrences(text: str, needle: str) -> List[Tuple[int, int]]:
    if not needle:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), text)]


def _normalize_variants(s: str) -> List[str]:
    variants = [s]
    variants.append(s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'"))
    variants.append(s.replace("\u00a0", " ").replace("\xa0", " "))
    out: List[str] = []
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
    ids: str = "",
    segment_prompt_path: Optional[str | Path] = None,
    tag_prompt_path: Optional[str | Path] = None,
) -> None:
    root = _project_root()

    p1, p2 = _resolve_pipeline_prompts(segment_prompt_path, tag_prompt_path)

    tmpl1 = load_prompt(p1)
    tmpl2 = load_prompt(p2)

    df = pd.read_csv(dataset_csv)
    text_col = _pick_text_column(df)

    if ids.strip():
        wanted = {int(x) for x in ids.split(",") if x.strip().isdigit()}
        df = df[df["id"].isin(wanted)].copy()

    outputs: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rid = int(row["id"])
        text = str(row[text_col])

        try:
            prompt1 = fill(tmpl1, REQUIREMENT_TEXT=text)
            out1 = ollama_generate(prompt1, model=model)
            clauses = out1.get("clauses", []) if isinstance(out1, dict) else []
            clauses = validate_clauses(text, clauses)

            prompt2 = fill(tmpl2, REQUIREMENT_TEXT=text, CLAUSES_JSON=json.dumps(clauses, ensure_ascii=False))
            out2 = ollama_generate(prompt2, model=model)
            spans = out2.get("spans", []) if isinstance(out2, dict) else []
            spans = validate_spans(text, spans)

            outputs.append({"id": rid, "text": text, "clauses": clauses, "spans": spans})
        except Exception as e:
            outputs.append({"id": rid, "text": text, "clauses": [], "spans": [], "error": str(e)})

    Path(out_json).write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} items to {out_json}")
