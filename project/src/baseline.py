from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .ollama import ollama_generate
from .common import (
    A3_TAGS,
    align_offsets,
    dedupe_spans,
    enforce_consistency,
    postprocess_spans,
    fill_template,
    load_prompt,
    normalize_tag,
    pick_by_variant,
)

def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # accept text or text_en
    if "text" not in df.columns and "text_en" in df.columns:
        df = df.rename(columns={"text_en": "text"})
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("Dataset must contain columns: id, text (or text_en).")
    return df

def run_baseline(
    dataset_csv: str | Path,
    ids: Optional[Sequence[int]] = None,
    *,
    variant: Optional[str] = None,
    model: Optional[str] = None,
    with_offsets: bool = True,
) -> List[Dict[str, Any]]:
    prompts_dir = Path(os.getenv("REQFLOW_PROMPTS_DIR", "prompts"))
    if not prompts_dir.is_absolute():
        prompts_dir = (Path(__file__).resolve().parents[1] / prompts_dir).resolve()

    v = (variant or os.getenv("REQFLOW_DEFAULT_BASELINE_VARIANT", "zero")).strip().lower()
    prompt_rel = pick_by_variant("REQFLOW_BASELINE_PROMPT", v)
    prompt = load_prompt(prompts_dir, prompt_rel)

    df = load_dataset(dataset_csv)
    if ids:
        ids_set = set(int(x) for x in ids)
        df = df[df["id"].astype(int).isin(ids_set)]

    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rid = int(row["id"])
        text = str(row["text"])

        full_prompt = fill_template(prompt, REQUIREMENT_TEXT=text, REQUIREMENT=text, REQ=text) if any(tok in prompt for tok in ("{REQUIREMENT_TEXT}","{{REQUIREMENT_TEXT}}","{REQUIREMENT}","{{REQUIREMENT}}","{REQ}","{{REQ}}")) else (prompt.strip()+"\n\nINPUT:\n"+text+"\n")
        obj = ollama_generate(full_prompt, model=model, retries=int(os.getenv("OLLAMA_RETRIES", "2")))

        spans_raw = obj.get("spans", [])
        spans: List[Dict[str, Any]] = []
        if isinstance(spans_raw, list):
            for sp in spans_raw:
                if not isinstance(sp, dict):
                    continue
                tag = normalize_tag(sp.get("tag"))
                seg = sp.get("text")
                if tag and isinstance(seg, str) and seg.strip():
                    spans.append({"tag": tag, "text": seg.strip()})

        # 1) normalize + subset consistency
        spans = enforce_consistency(dedupe_spans(spans))
        # 2) text-bound repair (punctuation/list boundaries) + per-tag caps
        spans = postprocess_spans(text, spans)

        if with_offsets:
            spans_vis = align_offsets(text, spans)
        else:
            spans_vis = spans

        out.append({"id": rid, "text": text, "spans": spans_vis})

    return out
