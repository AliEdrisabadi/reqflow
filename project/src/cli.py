from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

from env import load_dotenv, find_dotenv
from ollama import ollama_check

import baseline
import pipeline

# NOTE: render.py is optional.
import json
import html as _html


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

TAG_COLORS_DEFAULT = {
    "Main_actor": "#66c2ff",
    "Entity": "#a78bfa",
    "Action": "#2bd98f",
    "System_response": "#ffb020",
    "Condition": "#f472b6",
    "Precondition": "#60a5fa",
    "Constraint": "#fb7185",
    "Exception": "#f59e0b",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_path(root: Path, key: str, default_rel: str) -> Path:
    raw = (os.getenv(key) or "").strip()
    p = Path(raw) if raw else Path(default_rel)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _variant_to_key(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ("0", "zero", "zero-shot", "zeroshot"):
        return "zero"
    if v in ("1", "one", "one-shot", "oneshot"):
        return "one"
    if v in ("few", "few-shot", "fewshot", "kshot"):
        return "few"
    return "zero"


def _resolve_prompt_by_variant(root: Path, kind: str, variant: str) -> Path:
    prompts_dir = _env_path(root, "REQFLOW_PROMPTS_DIR", "prompts")
    v = _variant_to_key(variant)
    env_key = f"REQFLOW_{kind}_PROMPT_{v.upper()}"
    name = (os.getenv(env_key) or "").strip()
    if not name:
        raise FileNotFoundError(f"Missing {env_key} in .env (needed for {kind} {v}).")

    p = Path(name)
    if not p.is_absolute():
        p = prompts_dir / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found for {env_key}: {p}")
    return p


def _pct(x: float) -> str:
    return f"{x * 100.0:.2f}%"


def _bars_for_tag(text: str, spans, tag_colors) -> str:
    n = max(1, len(text))
    bars = []
    for sp in spans:
        st = int(sp.get("start", 0))
        en = int(sp.get("end", 0))
        tag = str(sp.get("tag", ""))
        seg = str(sp.get("text", ""))

        left = max(0.0, min(1.0, st / n))
        width = max(0.0, min(1.0, (en - st) / n))

        color = tag_colors.get(tag, "#999999")
        tip = _html.escape(f"{tag}: {seg}")
        bars.append(
            f'<div class="bar" style="left:{_pct(left)};width:{_pct(width)};background:{color};" title="{tip}"></div>'
        )
    return "\n".join(bars)


def _render_item(item, tag_colors) -> str:
    rid = item.get("id", "")
    text = str(item.get("text", ""))
    spans = item.get("spans", []) or []

    by_tag = {t: [] for t in TAGS}
    for sp in spans:
        tag = sp.get("tag")
        if tag in by_tag:
            by_tag[tag].append(sp)

    rows = []
    for tag in TAGS:
        bars = _bars_for_tag(text, by_tag[tag], tag_colors=tag_colors)
        rows.append(
            f"""
            <div class="tagRow">
              <div class="tagLabel">{_html.escape(tag)}</div>
              <div class="track">{bars}</div>
            </div>
            """
        )

    return f"""
    <section class="card">
      <div class="cardHeader">
        <div class="rid">ID {_html.escape(str(rid))}</div>
        <div class="reqText">{_html.escape(text)}</div>
      </div>
      <div class="rows">
        {''.join(rows)}
      </div>
    </section>
    """


def render_spans_html(pred_json: Path, out_html: Path, *, theme: str = "dark", tag_colors=None) -> None:
    items = json.loads(pred_json.read_text(encoding="utf-8"))
    tag_colors = tag_colors or dict(TAG_COLORS_DEFAULT)

    is_dark = (theme or "dark").lower().strip() != "light"
    bg = "#15181c" if is_dark else "#ffffff"
    card = "#0f1216" if is_dark else "#f7f7f7"
    textc = "#eaeaea" if is_dark else "#222222"
    mutec = "#a8b0bb" if is_dark else "#555555"
    border = "#2a2f36" if is_dark else "#e0e0e0"
    track = "#1b2026" if is_dark else "#ffffff"

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
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background: var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 1280px;
      margin: 16px auto;
      padding: 0 12px 28px 12px;
    }}
    .header {{
      display:flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin: 8px 0 18px 0;
    }}
    .header h1 {{
      font-size: 18px;
      margin: 0;
      font-weight: 800;
      letter-spacing: 0.2px;
    }}
    .header .meta {{
      font-size: 12px;
      color: var(--muted);
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      margin: 12px 0;
      box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }}
    .cardHeader {{
      margin-bottom: 10px;
    }}
    .rid {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
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
      grid-template-columns: 150px 1fr;
      gap: 10px;
      align-items: center;
    }}
    .tagLabel {{
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
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
      filter: saturate(1.1);
    }}
    """

    body = "\n".join(_render_item(it, tag_colors=tag_colors) for it in items)

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
            <div class="meta">{_html.escape(pred_json.name)}</div>
          </div>
          {body}
        </div>
      </body>
    </html>
    """
    out_html.write_text(html_doc, encoding="utf-8")


def run(mode: str, dataset: Path, out_dir: Path, ids: str = "", baseline_variant: str = "zero", pipeline_variant: str = "zero") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, msg = ollama_check(timeout=5)
    if not ok:
        raise SystemExit(msg)

    theme = os.getenv("REQFLOW_HTML_THEME", "dark").strip() or "dark"
    root = project_root()

    if mode in ("baseline", "both"):
        b_prompt = _resolve_prompt_by_variant(root, "BASELINE", baseline_variant)

        bdir = out_dir / "baseline" / _variant_to_key(baseline_variant)
        bdir.mkdir(parents=True, exist_ok=True)
        bj = bdir / "baseline_spans.json"
        bh = bdir / "baseline_spans.html"

        baseline.main(str(dataset), str(bj), model=None, prompt_path=str(b_prompt), ids=ids)
        render_spans_html(bj, bh, theme=theme)

    if mode in ("pipeline", "both"):
        s_prompt = _resolve_prompt_by_variant(root, "SEGMENT", pipeline_variant)
        t_prompt = _resolve_prompt_by_variant(root, "TAG", pipeline_variant)

        pdir = out_dir / "pipeline" / _variant_to_key(pipeline_variant)
        pdir.mkdir(parents=True, exist_ok=True)
        pj = pdir / "pipeline_spans.json"
        ph = pdir / "pipeline_spans.html"

        pipeline.main(
            str(dataset),
            str(pj),
            model=None,
            ids=ids,
            segment_prompt_path=str(s_prompt),
            tag_prompt_path=str(t_prompt),
        )
        render_spans_html(pj, ph, theme=theme)

    print(f"Done. Results in: {out_dir}")
    print(f"\n--- EVALUATION TIP ---")
    print(f"To evaluate results against gold annotations:")
    print(f"  python src/tools/evaluate.py --pred <output_json> --gold data/gold.json")
    print(f"\nRecommended evaluation: Relaxed Match with threshold=0.5")
    print(f"Primary metric: Micro F1 | Secondary metric: Macro F1")


def main() -> None:
    root = project_root()
    load_dotenv(find_dotenv(str(root)))

    ap = argparse.ArgumentParser(description="ReqFlow CLI runner (Ollama).")
    ap.add_argument("--dataset", default=str((root / "data" / "requirements_dataset.csv").resolve()))
    ap.add_argument("--mode", choices=["baseline", "pipeline", "both"], default="both")
    ap.add_argument("--ids", default="", help="Comma-separated requirement IDs")
    ap.add_argument("--outdir", default="", help="Output folder (default: result_cli/run_<timestamp>)")
    ap.add_argument("--baseline_variant", default=os.getenv("REQFLOW_DEFAULT_BASELINE_VARIANT", "zero"))
    ap.add_argument("--pipeline_variant", default=os.getenv("REQFLOW_DEFAULT_PIPELINE_VARIANT", "zero"))
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    out_dir = (
        Path(args.outdir).resolve()
        if args.outdir
        else (root / "result_cli" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    )
    run(args.mode, dataset, out_dir, ids=args.ids, baseline_variant=args.baseline_variant, pipeline_variant=args.pipeline_variant)


if __name__ == "__main__":
    main()
