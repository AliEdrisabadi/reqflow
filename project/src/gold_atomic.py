"""Gold atomicization utilities.

Why do this?
------------
Some gold spans are *composite* enumerations, e.g.:
    "seats by campus, date, and time range"

LLMs (especially multi-agent pipelines) often output more *atomic* pieces such as:
    "by campus", "date", "time range".

If evaluation compares only (tag, span_text) (ignoring offsets), composite gold
spans can create artificial mismatches. This module provides a conservative,
rule-based "atomicizer" to split list-like spans into smaller units.

Design goals
------------
- Preserve meaning as much as possible with simple heuristics.
- Prefer producing pieces that *actually appear* in the original requirement text
  so offsets can be recovered for HTML rendering if needed.
- Never change tags; only split / rewrite span texts.

NOTE: This is not "ground truth" rewriting; it is a pragmatic normalization step
for a text-only evaluation policy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Keep in sync with the course slide taxonomy (plus exact casing).
TAGS = [
    "Purpose",
    "Trigger",
    "Precondition",
    "Condition",
    "Action",
    "System_response",
    "Entity",
    "Main_actor",
]


_LIST_SPLIT_RE = re.compile(r"\s*,\s*|\s*;\s*")
_CONJ_SPLIT_RE = re.compile(r"\s+(?:and|or)\s+", re.IGNORECASE)


def _strip_outer_punct(s: str) -> str:
    return s.strip().strip(" \t\n\r\"'.,;:()[]{}")


def _find_offsets(full_text: str, span_text: str) -> Tuple[int, int]:
    """Best-effort case-insensitive search to recover (start,end) for span_text."""
    if not full_text or not span_text:
        return -1, -1
    m = re.search(re.escape(span_text), full_text, flags=re.IGNORECASE)
    if m:
        return m.start(), m.end()
    return -1, -1


def _atomicize_span_text(span_text: str) -> List[str]:
    """Split a span text into atomic pieces if it looks list-like.

    Heuristics:
    - Split on commas / semicolons.
    - Further split on conjunctions (and/or) when it looks like a flat list.
    - Normalize patterns like "X by Y" into "by Y" *when splitting*, because
      models tend to output the prepositional phrase.
    """
    s = span_text.strip()
    if not s:
        return []

    # Fast-path: no list indicators.
    if ("," not in s) and (";" not in s) and (" and " not in s.lower()) and (" or " not in s.lower()):
        return [span_text]

    # 1) split on commas/semicolons
    parts: List[str] = [p for p in _LIST_SPLIT_RE.split(s) if p and p.strip()]

    # 2) split further on conjunctions inside each part
    atomic: List[str] = []
    for p in parts:
        p = p.strip()
        # remove leading conjunction artifacts after comma split
        p = re.sub(r"^(and|or)\s+", "", p, flags=re.IGNORECASE).strip()
        if not p:
            continue
        subparts = [sp for sp in _CONJ_SPLIT_RE.split(p) if sp and sp.strip()]
        if len(subparts) > 1:
            atomic.extend([sp.strip() for sp in subparts if sp.strip()])
        else:
            atomic.append(p)

    # 3) "X by Y" -> "by Y" (only if it keeps something meaningful)
    out: List[str] = []
    for p in atomic:
        p2 = p.strip()
        m = re.match(r"^(.+?)\s+by\s+(.+)$", p2, flags=re.IGNORECASE)
        if m:
            tail = _strip_outer_punct(m.group(2))
            if tail:
                out.append(f"by {tail}")
                continue
        out.append(p2)

    # 4) final cleanup + dedup (preserve order)
    seen = set()
    cleaned: List[str] = []
    for p in out:
        p = _strip_outer_punct(p)
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(p)

    return cleaned if cleaned else [span_text]


def atomicize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of an item with atomicized spans."""
    text = item.get("text") or ""
    spans = item.get("spans") or []

    new_spans: List[Dict[str, Any]] = []
    for sp in spans:
        if not isinstance(sp, dict):
            continue
        tag = sp.get("tag")
        if tag not in TAGS:
            # keep unknown tags as-is
            new_spans.append(sp)
            continue
        st = sp.get("text")
        if not isinstance(st, str) or not st.strip():
            continue

        pieces = _atomicize_span_text(st)
        if len(pieces) == 1 and pieces[0].strip() == st.strip():
            # unchanged
            new_spans.append(sp)
            continue

        # replace composite span by its atomic pieces
        for piece in pieces:
            start, end = _find_offsets(text, piece)
            if start == -1:
                # fallback to original offsets if any
                start = int(sp.get("start", -1)) if str(sp.get("start", "")).lstrip("-").isdigit() else -1
                end = int(sp.get("end", -1)) if str(sp.get("end", "")).lstrip("-").isdigit() else -1
            new_spans.append({
                "tag": tag,
                "text": piece,
                "start": start,
                "end": end,
            })

    # Deduplicate within (tag, text) case-insensitive, preserve order
    seen2 = set()
    deduped: List[Dict[str, Any]] = []
    for sp in new_spans:
        tag = sp.get("tag")
        txt = sp.get("text")
        if not isinstance(tag, str) or not isinstance(txt, str):
            continue
        k = (tag, txt.strip().lower())
        if k in seen2:
            continue
        seen2.add(k)
        deduped.append(sp)

    out = dict(item)
    out["spans"] = deduped
    return out


def atomicize_gold_obj(gold: Dict[str, Any]) -> Dict[str, Any]:
    """Atomicize an in-memory gold object.

    Supports:
    - {"schema": [...], "items": [...]}
    - {"items": [...]} or {"data": [...]}
    - plain list of items
    """
    if isinstance(gold, list):
        items = gold
        schema = TAGS
        wrapper = "list"
    elif isinstance(gold, dict):
        if isinstance(gold.get("items"), list):
            items = gold["items"]
        elif isinstance(gold.get("data"), list):
            items = gold["data"]
        else:
            raise ValueError("Unsupported gold JSON format")
        schema = gold.get("schema") if isinstance(gold.get("schema"), list) else TAGS
        wrapper = "dict"
    else:
        raise ValueError("Unsupported gold JSON format")

    new_items = [atomicize_item(it) for it in items if isinstance(it, dict)]

    if wrapper == "list":
        return new_items

    out = dict(gold)
    out["schema"] = schema
    if "items" in out and isinstance(out["items"], list):
        out["items"] = new_items
    elif "data" in out and isinstance(out["data"], list):
        out["data"] = new_items
    else:
        out["items"] = new_items
    return out


def atomicize_gold_file(in_path: str, out_path: str) -> str:
    """Read a gold JSON, atomicize it, write to out_path. Returns out_path."""
    raw = json.loads(Path(in_path).read_text(encoding="utf-8"))
    out = atomicize_gold_obj(raw)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
