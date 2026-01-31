from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

A3_TAGS = [
    "Purpose",
    "Trigger",
    "Main_actor",
    "Entity",
    "Action",
    "System_response",
    "Condition",
    "Precondition",
]

TAG_ALIASES = {
    "Constraint": "Condition",
    "Exception": "Trigger",
}

def normalize_tag(tag: Any) -> Optional[str]:
    if not isinstance(tag, str):
        return None
    t = TAG_ALIASES.get(tag, tag)
    return t if t in A3_TAGS else None

def norm_text(s: str) -> str:
    # normalize whitespace + case for evaluation
    return " ".join((s or "").split()).strip().lower()

def load_prompt(prompts_dir: Path, rel: str) -> str:
    p = Path(rel)
    if not p.is_absolute():
        p = prompts_dir / p
    if not p.exists():
        raise FileNotFoundError(f"Prompt not found: {p}")
    return p.read_text(encoding="utf-8")

def pick_by_variant(prefix: str, variant: str) -> str:
    # prefix: e.g. REQFLOW_BASELINE_PROMPT or REQFLOW_AGENT_ENTITIES_PROMPT
    v = (variant or "zero").strip().lower()
    key = f"{prefix}_{v.upper()}"
    val = os.getenv(key, "").strip()
    if not val:
        raise KeyError(f"Missing env var: {key}")
    return val

def fill_template(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{{" + k + "}}", v)
        out = out.replace("{"+k+"}", v)  # tolerate {REQUIREMENT_TEXT}
    return out

def dedupe_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for sp in spans:
        tag = normalize_tag(sp.get("tag"))
        text = sp.get("text")
        if tag is None or not isinstance(text, str):
            continue
        key = (tag, text.strip())
        if key in seen:
            continue
        seen.add(key)
        sp2 = dict(sp)
        sp2["tag"] = tag
        sp2["text"] = text.strip()
        out.append(sp2)
    return out

def enforce_consistency(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # enforce subset rules by copying exact text spans
    by_tag = {t: set() for t in A3_TAGS}
    for sp in spans:
        t = normalize_tag(sp.get("tag"))
        if t and isinstance(sp.get("text"), str):
            by_tag[t].add(sp["text"])

    def add(tag: str, text: str):
        spans.append({"tag": tag, "text": text})

    # System_response ⊆ Action
    for txt in list(by_tag["System_response"]):
        if txt not in by_tag["Action"]:
            add("Action", txt)
            by_tag["Action"].add(txt)

    # Main_actor ⊆ Entity
    for txt in list(by_tag["Main_actor"]):
        if txt not in by_tag["Entity"]:
            add("Entity", txt)
            by_tag["Entity"].add(txt)

    # Precondition ⊆ Condition
    for txt in list(by_tag["Precondition"]):
        if txt not in by_tag["Condition"]:
            add("Condition", txt)
            by_tag["Condition"].add(txt)

    # Trigger ⊆ Condition
    for txt in list(by_tag["Trigger"]):
        if txt not in by_tag["Condition"]:
            add("Condition", txt)
            by_tag["Condition"].add(txt)

    return dedupe_spans(spans)

def align_offsets(text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Optional: compute start/end for visualization. Not used by evaluation.
    out = []
    for sp in spans:
        seg = sp.get("text")
        if not isinstance(seg, str) or not seg.strip():
            continue
        seg = seg.strip()

        # 1) exact match
        m = re.search(re.escape(seg), text)
        # 2) case-insensitive match
        if not m:
            m = re.search(re.escape(seg), text, flags=re.IGNORECASE)
        # 3) whitespace-normalized fuzzy (very light)
        if not m:
            seg_norm = " ".join(seg.split())
            txt_norm = " ".join(text.split())
            m2 = re.search(re.escape(seg_norm), txt_norm, flags=re.IGNORECASE)
            if m2:
                # mapping back to original indices is non-trivial; skip to avoid wrong offsets
                m = None

        sp2 = dict(sp)
        if m:
            sp2["start"] = int(m.start())
            sp2["end"] = int(m.end())
        out.append(sp2)
    return out
