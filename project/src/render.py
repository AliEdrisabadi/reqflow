from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List


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

# دقیقا همان پالت پیش‌فرض GUI
TAG_COLORS: Dict[str, str] = {
    "Main_actor": "#66c2ff",
    "Entity": "#a78bfa",
    "Action": "#2bd98f",
    "System_response": "#ffb020",
    "Condition": "#f472b6",
    "Precondition": "#60a5fa",
    "Constraint": "#fb7185",
    "Exception": "#f59e0b",
}


def _load(path: str | Path) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    return f"{x * 100.0:.2f}%"


def _bars_for_tag(text: str, spans: List[Dict[str, Any]]) -> str:
    n = max(1, len(text))
    out: List[str] = []

    for sp in spans or []:
        try:
            st = int(sp.get("start", 0))
            en = int(sp.get("end", 0))
        except Exception:
            continue

        if en < st:
            continue

        tag = str(sp.get("tag", "") or "")
        seg = str(sp.get("text", "") or "")

        # clamp
        st = max(0, min(st, n))
        en = max(0, min(en, n))

        left = st / n
        width = max(0.0, (en - st) / n)

        tip = html.escape(f"{tag}: {seg}")
        tag_class = f"tag_{tag}"  # underscore-safe

        out.append(
            f'<div class="bar {tag_class}" style="left:{_pct(left)};width:{_pct(width)};" title="{tip}"></div>'
        )

    return "\n".join(out)


def _render_item(item: Dict[str, Any]) -> str:
    rid = item.get("id", "")
    text = str(item.get("text", "") or "")
    spans = item.get("spans", []) or []

    by_tag: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TAGS}
    for sp in spans:
        tag = sp.get("tag")
        if tag in by_tag:
            by_tag[tag].append(sp)

    rows: List[str] = []
    for tag in TAGS:
        bars = _bars_for_tag(text, by_tag[tag])
        rows.append(
            f"""
            <div class="tagRow">
              <div class="tagLabel">{html.escape(tag)}</div>
              <div class="track">{bars}</div>
            </div>
            """
        )

    return f"""
    <section class="card">
      <div class="cardHeader">
        <div class="rid">ID {html.escape(str(rid))}</div>
        <div class="reqText">{html.escape(text)}</div>
      </div>
      <div class="rows">
        {''.join(rows)}
      </div>
    </section>
    """


def main(pred_json: str, out_html: str, *, theme: str = "dark") -> None:
    items = _load(pred_json)
    is_dark = (theme or "dark").lower().strip() != "light"

    # دقیقا مطابق GUI
    bg = "#15181c" if is_dark else "#ffffff"
    card = "#0f1216" if is_dark else "#f7f7f7"
    textc = "#eaeaea" if is_dark else "#222222"
    mutec = "#b9c3cf" if is_dark else "#555555"
    border = "#2a2f36" if is_dark else "#e0e0e0"
    track = "#1b2026" if is_dark else "#ffffff"

    # css classes per tag (هم‌راستا با GUI)
    tag_css = "\n".join(
        f".tag_{t} {{ background: {TAG_COLORS.get(t, '#999999')}; }}"
        for t in TAGS
    )

    css = f"""
    :root {{
      --bg: {bg};
      --card: {card};
      --text: {textc};
      --muted: {mutec};
      --border: {border};
      --track: {track};
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", ui-sans-serif, system-ui, -apple-system, Roboto, Arial;
      font-size: 12px;
    }}

    .container {{
      max-width: 1280px;
      margin: 14px auto;
      padding: 0 14px 24px 14px;
    }}

    .header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 14px;
      margin: 8px 0 14px 0;
    }}

    .header h1 {{
      font-size: 16px;
      margin: 0;
      font-weight: 900;
      letter-spacing: 0.2px;
      color: #f2f6fb;
    }}

    .header .meta {{
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
    }}

    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      margin: 12px 0;
      box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }}

    .rid {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      font-weight: 800;
    }}

    .reqText {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }}

    .rows {{
      margin-top: 12px;
      display: grid;
      gap: 8px;
    }}

    .tagRow {{
      display: grid;
      grid-template-columns: 160px 1fr;
      gap: 10px;
      align-items: center;
    }}

    .tagLabel {{
      font-size: 12px;
      color: var(--muted);
      font-weight: 800;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .track {{
      position: relative;
      height: 18px;
      border-radius: 10px;
      background: var(--track);
      border: 1px solid var(--border);
      overflow: hidden;
    }}

    .bar {{
      position: absolute;
      top: 1px;
      bottom: 1px;
      border-radius: 9px;
      opacity: 0.92;
    }}

    .bar:hover {{
      opacity: 1.0;
      filter: saturate(1.10);
    }}

    {tag_css}
    """

    body = "\n".join(_render_item(it) for it in items)

    html_doc = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>ReqFlow — Spans</title>
    <style>{css}</style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h1>ReqFlow — Span Visualization</h1>
        <div class="meta">{html.escape(Path(pred_json).name)}</div>
      </div>
      {body}
    </div>
  </body>
</html>
"""

    Path(out_html).write_text(html_doc, encoding="utf-8")
    print(f"Wrote HTML to {out_html}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_html", required=True)
    ap.add_argument("--theme", default="dark")
    args = ap.parse_args()

    main(args.in_json, args.out_html, theme=args.theme)
