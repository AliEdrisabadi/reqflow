from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .common import A3_TAGS, normalize_tag

TAG_COLORS: Dict[str, str] = {
    "Purpose": "#fb7185",
    "Trigger": "#f59e0b",
    "Main_actor": "#66c2ff",
    "Entity": "#a78bfa",
    "Action": "#2bd98f",
    "System_response": "#ffb020",
    "Condition": "#f472b6",
    "Precondition": "#60a5fa",
}

def load_items(path: str | Path) -> List[Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [it for it in raw if isinstance(it, dict)]
    if isinstance(raw, dict):
        if isinstance(raw.get("items"), list):
            return [it for it in raw["items"] if isinstance(it, dict)]
        if isinstance(raw.get("data"), list):
            return [it for it in raw["data"] if isinstance(it, dict)]
    raise ValueError("Unsupported JSON format for rendering.")

def _find_offset(text: str, seg: str) -> tuple[int,int] | None:
    if not seg:
        return None
    m = re.search(re.escape(seg), text)
    if not m:
        m = re.search(re.escape(seg), text, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.start()), int(m.end())

def _bars_for_tag(text: str, spans: List[Dict[str, Any]]) -> str:
    n = max(1, len(text))
    out: List[str] = []
    for sp in spans or []:
        tag = normalize_tag(sp.get("tag")) or ""
        seg = str(sp.get("text","") or "")
        st = sp.get("start")
        en = sp.get("end")
        if not (isinstance(st,int) and isinstance(en,int)):
            off = _find_offset(text, seg)
            if off:
                st,en = off
            else:
                continue
        st = max(0, min(int(st), n))
        en = max(0, min(int(en), n))
        if en < st:
            continue
        left = st / n
        width = max(0.0, (en - st) / n)
        tip = html.escape(f"{tag}: {seg}")
        cls = f"tag_{tag}"
        out.append(f'<div class="bar {cls}" style="left:{left*100:.2f}%;width:{width*100:.2f}%;" title="{tip}"></div>')
    return "\n".join(out)

def _render_item(item: Dict[str, Any]) -> str:
    rid = item.get("id","")
    text = str(item.get("text","") or "")
    spans = item.get("spans", []) or []
    by_tag: Dict[str, List[Dict[str, Any]]] = {t: [] for t in A3_TAGS}
    for sp in spans:
        if not isinstance(sp, dict):
            continue
        tag = normalize_tag(sp.get("tag"))
        if tag in by_tag:
            sp2 = dict(sp)
            sp2["tag"] = tag
            by_tag[tag].append(sp2)

    rows = []
    for tag in A3_TAGS:
        bars = _bars_for_tag(text, by_tag[tag])
        rows.append(f'''
        <div class="tagRow">
          <div class="tagLabel">{html.escape(tag)}</div>
          <div class="track">{bars}</div>
        </div>''')

    return f'''
    <section class="card">
      <div class="rid">ID {html.escape(str(rid))}</div>
      <div class="reqText">{html.escape(text)}</div>
      <div class="rows">
        {''.join(rows)}
      </div>
    </section>'''

def main(pred_json: str, out_html: str, *, theme: str = "dark") -> None:
    items = load_items(pred_json)
    is_dark = (theme or "dark").lower().strip() != "light"

    bg = "#15181c" if is_dark else "#ffffff"
    card = "#0f1216" if is_dark else "#f7f7f7"
    textc = "#eaeaea" if is_dark else "#222222"
    mutec = "#b9c3cf" if is_dark else "#555555"
    border = "#2a2f36" if is_dark else "#e0e0e0"
    track = "#1b2026" if is_dark else "#ffffff"

    tag_css = "\n".join(f".tag_{t} {{ background: {TAG_COLORS.get(t,'#999')}; }}" for t in A3_TAGS)

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>ReqFlow spans</title>
<style>
body {{ margin:0; background:{bg}; color:{textc}; font-family: ui-sans-serif, system-ui; font-size:12px; }}
.container {{ max-width:1280px; margin:14px auto; padding:0 14px 24px 14px; }}
.card {{ background:{card}; border:1px solid {border}; border-radius:12px; padding:14px; margin:12px 0; }}
.rid {{ color:{mutec}; font-weight:800; margin-bottom:6px; }}
.reqText {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size:13px; line-height:1.45; white-space: pre-wrap; }}
.rows {{ margin-top:12px; display:grid; gap:8px; }}
.tagRow {{ display:grid; grid-template-columns: 160px 1fr; gap:10px; align-items:center; }}
.tagLabel {{ font-weight:800; color:{mutec}; }}
.track {{ position:relative; height:14px; border:1px solid {border}; border-radius:9px; background:{track}; overflow:hidden; }}
.bar {{ position:absolute; top:0; bottom:0; border-radius:9px; opacity:0.95; }}
{tag_css}
</style>
</head>
<body>
<div class="container">
  <h2 style="margin:0 0 10px 0;">ReqFlow — Span Visualization</h2>
  <div style="color:{mutec}; margin-bottom:10px;">File: {html.escape(str(pred_json))}</div>
  {''.join(_render_item(it) for it in items)}
</div>
</body>
</html>"""

    Path(out_html).write_text(html_doc, encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--theme", default="dark")
    args = ap.parse_args()
    main(args.pred, args.out, theme=args.theme)
