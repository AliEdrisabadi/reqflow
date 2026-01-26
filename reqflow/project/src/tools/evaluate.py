"""
ReqFlow Evaluation Script

Computes evaluation metrics for span annotation:
- Supports Exact and Relaxed matching modes
- Reports both Micro F1 and Macro F1
- Configurable threshold for Relaxed mode

Usage:
    python evaluate.py --pred predictions.json --gold gold.json --out results.csv
    python evaluate.py --pred predictions.json --gold gold.json --mode relaxed --threshold 0.5
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


@dataclass(frozen=True)
class Span:
    tag: str
    start: int
    end: int
    text: str


def load_items(path: str) -> List[Dict[str, Any]]:
    """Load items from JSON file (supports multiple formats)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(raw, list):
        return [it for it in raw if isinstance(it, dict)]

    if isinstance(raw, dict):
        if isinstance(raw.get("items"), list):
            return [it for it in raw["items"] if isinstance(it, dict)]
        if isinstance(raw.get("data"), list):
            return [it for it in raw["data"] if isinstance(it, dict)]

        out: List[Dict[str, Any]] = []
        for _k, v in raw.items():
            if isinstance(v, dict):
                out.append(v)
        if out:
            return out

    raise ValueError("Unsupported JSON format (expected list or dict with items/data).")


def index_by_id(items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Index items by their ID field."""
    out: Dict[int, Dict[str, Any]] = {}
    for it in items:
        if "id" not in it:
            continue
        try:
            out[int(it["id"])] = it
        except Exception:
            continue
    return out


def _to_spans(item: dict) -> List[Span]:
    """Convert item spans to Span objects."""
    spans: List[Span] = []
    for sp in (item.get("spans", []) or []):
        tag = sp.get("tag")
        st_i = sp.get("start")
        en_i = sp.get("end")
        tx = sp.get("text", "")
        if tag in TAGS and isinstance(st_i, int) and isinstance(en_i, int) and en_i >= st_i:
            spans.append(Span(tag=tag, start=st_i, end=en_i, text=str(tx)))
    return spans


def to_set(spans: Any):
    """Convert spans to set of (tag, start, end) tuples for exact matching."""
    s = set()
    for sp in spans or []:
        if not isinstance(sp, dict):
            continue
        tag = sp.get("tag")
        st = sp.get("start")
        en = sp.get("end")
        if tag in TAGS and isinstance(st, int) and isinstance(en, int):
            s.add((tag, st, en))
    return s


# ============================================
# Relaxed Matching Functions
# ============================================

def iou(a: Span, b: Span) -> float:
    """Calculate Intersection over Union for two spans."""
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    if inter <= 0:
        return 0.0
    la = max(0, a.end - a.start)
    lb = max(0, b.end - b.start)
    union = la + lb - inter
    return (inter / union) if union > 0 else 0.0


def text_sim(a: Span, b: Span) -> float:
    """Calculate text similarity using SequenceMatcher."""
    return SequenceMatcher(None, a.text, b.text).ratio()


def span_score(a: Span, b: Span, mode: str) -> float:
    """
    Calculate match score between two spans.
    
    For Exact mode: 1.0 if exact match, 0.0 otherwise
    For Relaxed mode: 0.65 * IoU + 0.35 * TextSimilarity
    """
    if a.tag != b.tag:
        return 0.0
    
    if mode.lower() == "exact":
        return 1.0 if (a.start == b.start and a.end == b.end) else 0.0
    
    # Relaxed mode
    s_iou = iou(a, b)
    if s_iou <= 0:
        return 0.0
    s_txt = text_sim(a, b)
    
    return 0.65 * s_iou + 0.35 * s_txt


def match_counts(pred: List[Span], gold: List[Span], mode: str, threshold: float) -> Tuple[int, int, int]:
    """
    Greedy matching algorithm to compute TP, FP, FN.
    
    1. Calculate scores for all pred-gold pairs
    2. Filter pairs where score >= threshold
    3. Sort by score (highest first)
    4. Greedily assign matches (each span matched at most once)
    """
    cands: List[Tuple[float, int, int]] = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            sc = span_score(p, g, mode=mode)
            if sc >= threshold:
                cands.append((sc, i, j))

    cands.sort(key=lambda x: x[0], reverse=True)
    
    used_p = set()
    used_g = set()
    tp = 0
    
    for _sc, i, j in cands:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1

    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculate Precision, Recall, F1."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
    return p, r, f1


# ============================================
# Main Evaluation Functions
# ============================================

def evaluate_exact(pred: Dict[int, dict], gold: Dict[int, dict], common: List[int]) -> Tuple[dict, List[dict]]:
    """Exact match evaluation (original behavior)."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for rid in common:
        ps = to_set(pred[rid].get("spans"))
        gs = to_set(gold[rid].get("spans"))

        for tag in TAGS:
            p_tag = {x for x in ps if x[0] == tag}
            g_tag = {x for x in gs if x[0] == tag}
            counts[tag]["tp"] += len(p_tag & g_tag)
            counts[tag]["fp"] += len(p_tag - g_tag)
            counts[tag]["fn"] += len(g_tag - p_tag)
    
    return _compute_metrics(counts)


def evaluate_relaxed(pred: Dict[int, dict], gold: Dict[int, dict], common: List[int], threshold: float) -> Tuple[dict, List[dict]]:
    """Relaxed match evaluation with IoU + text similarity."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for rid in common:
        pred_spans = _to_spans(pred[rid])
        gold_spans = _to_spans(gold[rid])
        
        for tag in TAGS:
            p_tag = [s for s in pred_spans if s.tag == tag]
            g_tag = [s for s in gold_spans if s.tag == tag]
            
            tp, fp, fn = match_counts(p_tag, g_tag, mode="relaxed", threshold=threshold)
            counts[tag]["tp"] += tp
            counts[tag]["fp"] += fp
            counts[tag]["fn"] += fn
    
    return _compute_metrics(counts)


def _compute_metrics(counts: dict) -> Tuple[dict, List[dict]]:
    """Compute Micro and Macro F1 from per-tag counts."""
    rows = []
    tag_f1_scores = []
    overall_tp = overall_fp = overall_fn = 0
    
    for tag in TAGS:
        tp = counts[tag]["tp"]
        fp = counts[tag]["fp"]
        fn = counts[tag]["fn"]
        p, r, f1 = prf(tp, fp, fn)
        tag_f1_scores.append(f1)
        
        rows.append({
            "tag": tag,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        })
        
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
    
    # Micro F1
    p_micro, r_micro, f1_micro = prf(overall_tp, overall_fp, overall_fn)
    
    # Macro F1
    f1_macro = sum(tag_f1_scores) / len(tag_f1_scores) if tag_f1_scores else 0.0
    
    summary = {
        "tp": overall_tp,
        "fp": overall_fp,
        "fn": overall_fn,
        "precision": p_micro,
        "recall": r_micro,
        "micro_f1": f1_micro,
        "macro_f1": f1_macro,
    }
    
    return summary, rows


def main(pred_path: str, gold_path: str, out_csv: str, mode: str = "relaxed", threshold: float = 0.5):
    """
    Main evaluation function.
    
    Args:
        pred_path: Path to predictions JSON
        gold_path: Path to gold standard JSON
        out_csv: Output CSV path
        mode: "exact" or "relaxed" (default: relaxed)
        threshold: Match threshold for relaxed mode (default: 0.5)
    """
    pred = index_by_id(load_items(pred_path))
    gold = index_by_id(load_items(gold_path))

    common = sorted(set(pred.keys()) & set(gold.keys()))
    if not common:
        raise RuntimeError("No common IDs between pred and gold.")

    print(f"Evaluating {len(common)} common requirements...")
    print(f"Mode: {mode.upper()}, Threshold: {threshold}")
    print("-" * 60)
    
    # Run evaluation based on mode
    if mode.lower() == "exact":
        summary, rows = evaluate_exact(pred, gold, common)
    else:
        summary, rows = evaluate_relaxed(pred, gold, common, threshold)
    
    # Also compute exact match for comparison if in relaxed mode
    if mode.lower() != "exact":
        exact_summary, _ = evaluate_exact(pred, gold, common)
        exact_f1 = exact_summary["micro_f1"]
    else:
        exact_f1 = summary["micro_f1"]
    
    # Build CSV output
    lines = ["tag,tp,fp,fn,precision,recall,f1"]
    for row in rows:
        lines.append(
            f"{row['tag']},{row['tp']},{row['fp']},{row['fn']},"
            f"{row['precision']:.4f},{row['recall']:.4f},{row['f1']:.4f}"
        )
    
    # Add summary rows
    lines.append("")
    lines.append(f"# Summary")
    lines.append(f"# Mode: {mode.upper()}")
    lines.append(f"# Threshold: {threshold}")
    lines.append(f"# Common requirements: {len(common)}")
    lines.append(f"# TP: {summary['tp']}, FP: {summary['fp']}, FN: {summary['fn']}")
    lines.append(f"MICRO_F1,,,,,, {summary['micro_f1']:.4f}")
    lines.append(f"MACRO_F1,,,,,, {summary['macro_f1']:.4f}")
    if mode.lower() != "exact":
        lines.append(f"EXACT_F1,,,,,, {exact_f1:.4f}")

    Path(out_csv).write_text("\n".join(lines), encoding="utf-8")
    
    # Print summary to console
    print(f"\nResults:")
    print(f"  Micro Precision: {summary['precision']:.4f}")
    print(f"  Micro Recall:    {summary['recall']:.4f}")
    print(f"  Micro F1:        {summary['micro_f1']:.4f} (PRIMARY)")
    print(f"  Macro F1:        {summary['macro_f1']:.4f}")
    if mode.lower() != "exact":
        print(f"  Exact Match F1:  {exact_f1:.4f} (baseline)")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate span annotation predictions against gold standard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Relaxed evaluation (recommended)
  python evaluate.py --pred predictions.json --gold gold.json --mode relaxed --threshold 0.5

  # Exact match evaluation
  python evaluate.py --pred predictions.json --gold gold.json --mode exact

  # Stricter relaxed evaluation
  python evaluate.py --pred predictions.json --gold gold.json --mode relaxed --threshold 0.8
        """
    )
    ap.add_argument("--pred", required=True, help="Path to predictions JSON file")
    ap.add_argument("--gold", required=True, help="Path to gold standard JSON file")
    ap.add_argument("--out", default="results_f1.csv", help="Output CSV file (default: results_f1.csv)")
    ap.add_argument("--mode", choices=["exact", "relaxed"], default="relaxed",
                    help="Matching mode: 'exact' or 'relaxed' (default: relaxed)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Match threshold for relaxed mode (default: 0.5, recommended range: 0.3-0.8)")
    
    args = ap.parse_args()
    main(args.pred, args.gold, args.out, mode=args.mode, threshold=args.threshold)
