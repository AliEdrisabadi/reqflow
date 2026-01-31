from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.env import load_dotenv
from src.baseline import run_baseline
from src.pipeline import run_pipeline
from src.render import main as render_html

def _parse_ids(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None

def _default_out_dir() -> Path:
    base = Path(os.getenv("REQFLOW_RESULTS_DIR", "results"))
    if not base.is_absolute():
        base = (Path(__file__).resolve().parent / base).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out

def main():
    load_dotenv()

    ap = argparse.ArgumentParser(prog="reqflow")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser("baseline", help="Run single-agent baseline.")
    p_base.add_argument("--dataset", default=os.getenv("REQFLOW_DATASET", "data/dataset.csv"))
    p_base.add_argument("--variant", choices=["zero","one","few"], default=os.getenv("REQFLOW_DEFAULT_BASELINE_VARIANT","zero"))
    p_base.add_argument("--ids", default=None, help="Comma-separated requirement IDs.")
    p_base.add_argument("--model", default=None)
    p_base.add_argument("--out", default=None, help="Output JSON path.")

    p_pipe = sub.add_parser("pipeline", help="Run multi-agent pipeline (segmenter + agents).")
    p_pipe.add_argument("--dataset", default=os.getenv("REQFLOW_DATASET", "data/dataset.csv"))
    p_pipe.add_argument("--variant", choices=["zero","one","few"], default=os.getenv("REQFLOW_DEFAULT_PIPELINE_VARIANT","zero"))
    p_pipe.add_argument("--ids", default=None)
    p_pipe.add_argument("--model", default=None)
    p_pipe.add_argument("--out", default=None)

    p_render = sub.add_parser("render", help="Render prediction JSON to HTML.")
    p_render.add_argument("--pred", required=True)
    p_render.add_argument("--out", required=True)
    p_render.add_argument("--theme", default=os.getenv("REQFLOW_HTML_THEME","dark"))

    sub.add_parser("example", help="Print example commands.")

    args = ap.parse_args()

    if args.cmd == "example":
        print("Examples:")
        print("  python cli.py baseline --variant one --ids 1,2,3")
        print("  python cli.py pipeline --variant one --ids 1,2,3")
        print("  python evaluate.py --pred results/pipeline.json --gold data/gold.json --out results.csv")
        return

    if args.cmd == "baseline":
        ids = _parse_ids(args.ids)
        items = run_baseline(args.dataset, ids, variant=args.variant, model=args.model, with_offsets=True)
        out_dir = _default_out_dir()
        out_path = Path(args.out) if args.out else (out_dir / f"baseline_{args.variant}.json")
        out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        # auto-render
        html_path = out_path.with_suffix(".html")
        render_html(str(out_path), str(html_path), theme=os.getenv("REQFLOW_HTML_THEME","dark"))
        print(f"Wrote {html_path}")

    if args.cmd == "pipeline":
        ids = _parse_ids(args.ids)
        items = run_pipeline(args.dataset, ids, variant=args.variant, model=args.model, with_offsets=True)
        out_dir = _default_out_dir()
        out_path = Path(args.out) if args.out else (out_dir / f"pipeline_{args.variant}.json")
        out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        html_path = out_path.with_suffix(".html")
        render_html(str(out_path), str(html_path), theme=os.getenv("REQFLOW_HTML_THEME","dark"))
        print(f"Wrote {html_path}")

    if args.cmd == "render":
        render_html(args.pred, args.out, theme=args.theme)

if __name__ == "__main__":
    main()
