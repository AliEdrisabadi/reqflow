from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from env import load_dotenv, find_dotenv
from ollama import ollama_check
import baseline
import pipeline
import render


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(mode: str, dataset: Path, out_dir: Path, ids: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, msg = ollama_check(timeout=5)
    if not ok:
        raise SystemExit(msg)

    theme = "dark"

    if mode in ("baseline", "both"):
        bj = out_dir / "baseline_spans.json"
        bh = out_dir / "baseline_spans.html"
        baseline.main(str(dataset), str(bj), model=None, prompt_path=None, ids=ids)
        render.main(str(bj), str(bh), theme=theme)

    if mode in ("pipeline", "both"):
        pj = out_dir / "pipeline_spans.json"
        ph = out_dir / "pipeline_spans.html"
        pipeline.main(str(dataset), str(pj), model=None, ids=ids)
        render.main(str(pj), str(ph), theme=theme)

    print(f"Done. Results in: {out_dir}")


def main() -> None:
    root = project_root()
    load_dotenv(find_dotenv(str(root)))

    ap = argparse.ArgumentParser(description="ReqFlow CLI runner (Ollama).")
    ap.add_argument("--dataset", default=str((root / "data" / "requirements_dataset.csv").resolve()))
    ap.add_argument("--mode", choices=["baseline", "pipeline", "both"], default="both")
    ap.add_argument("--ids", default="", help="Comma-separated requirement IDs")
    ap.add_argument("--outdir", default="", help="Output folder (default: result_cli/run_<timestamp>)")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    out_dir = (
        Path(args.outdir).resolve()
        if args.outdir
        else (root / "result_cli" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    )
    run(args.mode, dataset, out_dir, ids=args.ids)


if __name__ == "__main__":
    main()
