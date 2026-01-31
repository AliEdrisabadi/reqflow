from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .ollama import ollama_generate
from .common import (
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
    if "text" not in df.columns and "text_en" in df.columns:
        df = df.rename(columns={"text_en": "text"})
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("Dataset must contain columns: id, text (or text_en).")
    return df

def _get_prompts_dir() -> Path:
    prompts_dir = Path(os.getenv("REQFLOW_PROMPTS_DIR", "prompts"))
    if not prompts_dir.is_absolute():
        prompts_dir = (Path(__file__).resolve().parents[1] / prompts_dir).resolve()
    return prompts_dir

def _call_agent(prompt_text: str, input_text: str, *, model: Optional[str]) -> Dict[str, Any]:
    # Some prompts may not include placeholders; in that case append the input explicitly.
    if any(tok in prompt_text for tok in (
        "{REQUIREMENT_TEXT}", "{{REQUIREMENT_TEXT}}",
        "{REQUIREMENT}", "{{REQUIREMENT}}",
        "{REQ}", "{{REQ}}",
    )):
        full_prompt = fill_template(
            prompt_text,
            REQUIREMENT_TEXT=input_text,
            REQUIREMENT=input_text,
            REQ=input_text,
        )
    else:
        full_prompt = prompt_text.strip() + "\n\nINPUT:\n" + input_text + "\n"
    return ollama_generate(full_prompt, model=model, retries=int(os.getenv("OLLAMA_RETRIES", "2")))

def run_pipeline(
    dataset_csv: str | Path,
    ids: Optional[Sequence[int]] = None,
    *,
    variant: Optional[str] = None,
    model: Optional[str] = None,
    with_offsets: bool = True,
) -> List[Dict[str, Any]]:
    prompts_dir = _get_prompts_dir()
    v = (variant or os.getenv("REQFLOW_DEFAULT_PIPELINE_VARIANT", "zero")).strip().lower()

    # load agent prompts by variant
    segment_rel = pick_by_variant("REQFLOW_AGENT_SEGMENT_PROMPT", v)
    ent_rel = pick_by_variant("REQFLOW_AGENT_ENTITIES_PROMPT", v)
    act_rel = pick_by_variant("REQFLOW_AGENT_ACTIONS_PROMPT", v)
    logic_rel = pick_by_variant("REQFLOW_AGENT_LOGIC_PROMPT", v)
    purp_rel = pick_by_variant("REQFLOW_AGENT_PURPOSE_PROMPT", v)

    p_segment = load_prompt(prompts_dir, segment_rel)
    p_entities = load_prompt(prompts_dir, ent_rel)
    p_actions = load_prompt(prompts_dir, act_rel)
    p_logic = load_prompt(prompts_dir, logic_rel)
    p_purpose = load_prompt(prompts_dir, purp_rel)

    use_segmenter = os.getenv("REQFLOW_PIPELINE_USE_SEGMENTER", "true").strip().lower() in ("1","true","yes","y")

    df = load_dataset(dataset_csv)
    if ids:
        ids_set = set(int(x) for x in ids)
        df = df[df["id"].astype(int).isin(ids_set)]

    out: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        rid = int(row["id"])
        full_text = str(row["text"])

        segments = [full_text]
        if use_segmenter:
            # Heuristic: most dataset items are single-sentence and short.
            # Segmenter can inadvertently break comma-separated criteria (hurting span boundaries).
            # So we only segment if there are 2+ sentence-like units.
            sentence_like = len([s for s in re.split(r"[.!?]", full_text) if s.strip()])
            if sentence_like > 1:
                try:
                    seg_obj = _call_agent(p_segment, full_text, model=model)
                    segs = seg_obj.get("segments")
                    if isinstance(segs, list):
                        segs2 = [s for s in segs if isinstance(s, str) and s.strip()]
                        # keep only exact-substring segments
                        segs2 = [s for s in segs2 if (s in full_text) or (s.lower() in full_text.lower())]
                        if 1 <= len(segs2) <= 6:
                            segments = segs2
                except Exception:
                    segments = [full_text]

        # Ensure segments are unique and preserve original order
        seg_seen = set()
        segments = [s for s in segments if not (s in seg_seen or seg_seen.add(s))]

        all_spans: List[Dict[str, Any]] = []

        for seg in segments:
            # Entities agent
            ent_obj = _call_agent(p_entities, seg, model=model)
            for tag_key in ("Entity", "Main_actor"):
                arr = ent_obj.get(tag_key, [])
                if isinstance(arr, list):
                    for s in arr:
                        if isinstance(s, str) and s.strip():
                            all_spans.append({"tag": tag_key, "text": s.strip()})

            # Actions agent
            act_obj = _call_agent(p_actions, seg, model=model)
            for tag_key in ("Action", "System_response"):
                arr = act_obj.get(tag_key, [])
                if isinstance(arr, list):
                    for s in arr:
                        if isinstance(s, str) and s.strip():
                            all_spans.append({"tag": tag_key, "text": s.strip()})

            # Logic agent
            logic_obj = _call_agent(p_logic, seg, model=model)
            for tag_key in ("Condition", "Precondition", "Trigger"):
                arr = logic_obj.get(tag_key, [])
                if isinstance(arr, list):
                    for s in arr:
                        if isinstance(s, str) and s.strip():
                            all_spans.append({"tag": tag_key, "text": s.strip()})

            # Purpose agent
            purp_obj = _call_agent(p_purpose, seg, model=model)
            arr = purp_obj.get("Purpose", [])
            if isinstance(arr, list):
                for s in arr:
                    if isinstance(s, str) and s.strip():
                        all_spans.append({"tag": "Purpose", "text": s.strip()})

        # normalize + drop non A3
        spans_norm: List[Dict[str, Any]] = []
        for sp in all_spans:
            tag = normalize_tag(sp.get("tag"))
            txt = sp.get("text")
            if tag and isinstance(txt, str) and txt.strip():
                if (txt.strip() in full_text) or (txt.strip().lower() in full_text.lower()):  # keep substrings only
                    spans_norm.append({"tag": tag, "text": txt.strip()})

        spans_norm = enforce_consistency(dedupe_spans(spans_norm))
        spans_norm = postprocess_spans(full_text, spans_norm)

        if with_offsets:
            spans_vis = align_offsets(full_text, spans_norm)
        else:
            spans_vis = spans_norm

        out.append({"id": rid, "text": full_text, "spans": spans_vis})

    return out
