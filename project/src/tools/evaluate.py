import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
from difflib import SequenceMatcher


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


def load_items(path: str) -> List[Dict[str, Any]]:
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
    out: Dict[int, Dict[str, Any]] = {}
    for it in items:
        if "id" not in it:
            continue
        try:
            out[int(it["id"])] = it
        except Exception:
            continue
    return out


def to_span_list(spans: Any) -> List[Dict]:
    """Convert spans to list of dicts with tag, start, end, text."""
    result = []
    for sp in spans or []:
        if not isinstance(sp, dict):
            continue
        tag = sp.get("tag")
        st = sp.get("start")
        en = sp.get("end")
        text = sp.get("text", "")
        if tag in TAGS and isinstance(st, int) and isinstance(en, int):
            result.append({"tag": tag, "start": st, "end": en, "text": text})
    return result


def to_set(spans: Any) -> Set[Tuple[str, int, int]]:
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


def compute_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """Compute Intersection over Union for two spans."""
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)
    
    len1 = end1 - start1
    len2 = end2 - start2
    union = len1 + len2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute text similarity using SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def is_relaxed_match(pred_span: Dict, gold_span: Dict, threshold: float) -> bool:
    """
    Check if predicted span matches gold span using relaxed criteria.
    Match if: IoU >= threshold OR TextSimilarity >= threshold
    """
    # Must have same tag
    if pred_span["tag"] != gold_span["tag"]:
        return False
    
    # Compute IoU
    iou = compute_iou(
        pred_span["start"], pred_span["end"],
        gold_span["start"], gold_span["end"]
    )
    
    if iou >= threshold:
        return True
    
    # Compute text similarity as fallback
    text_sim = compute_text_similarity(pred_span["text"], gold_span["text"])
    
    return text_sim >= threshold


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute Precision, Recall, F1."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate_exact(pred: Dict, gold: Dict, common: List[int]) -> Dict[str, Dict[str, int]]:
    """Exact match evaluation."""
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
    
    return counts


def evaluate_relaxed(pred: Dict, gold: Dict, common: List[int], threshold: float) -> Dict[str, Dict[str, int]]:
    """Relaxed match evaluation with IoU/TextSimilarity threshold."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for rid in common:
        pred_spans = to_span_list(pred[rid].get("spans"))
        gold_spans = to_span_list(gold[rid].get("spans"))
        
        for tag in TAGS:
            p_tag = [s for s in pred_spans if s["tag"] == tag]
            g_tag = [s for s in gold_spans if s["tag"] == tag]
            
            # Track which gold spans have been matched
            matched_gold = set()
            matched_pred = set()
            
            # For each predicted span, find best matching gold span
            for pi, p_span in enumerate(p_tag):
                best_match = -1
                best_score = 0.0
                
                for gi, g_span in enumerate(g_tag):
                    if gi in matched_gold:
                        continue
                    
                    iou = compute_iou(p_span["start"], p_span["end"], 
                                     g_span["start"], g_span["end"])
                    text_sim = compute_text_similarity(p_span["text"], g_span["text"])
                    score = max(iou, text_sim)
                    
                    if score >= threshold and score > best_score:
                        best_score = score
                        best_match = gi
                
                if best_match >= 0:
                    matched_gold.add(best_match)
                    matched_pred.add(pi)
            
            tp = len(matched_pred)
            fp = len(p_tag) - tp
            fn = len(g_tag) - len(matched_gold)
            
            counts[tag]["tp"] += tp
            counts[tag]["fp"] += fp
            counts[tag]["fn"] += fn
    
    return counts


def main(pred_path: str, gold_path: str, out_csv: str, mode: str = "exact", threshold: float = 0.5):
    """
    Evaluate predictions against gold annotations.
    
    Args:
        pred_path: Path to predictions JSON
        gold_path: Path to gold annotations JSON
        out_csv: Output CSV path
        mode: "exact" or "relaxed"
        threshold: IoU/TextSimilarity threshold for relaxed matching (default: 0.5)
    
    Computes per-tag metrics and overall Micro F1 (primary) and Macro F1 (secondary).
    """
    pred = index_by_id(load_items(pred_path))
    gold = index_by_id(load_items(gold_path))

    common = sorted(set(pred.keys()) & set(gold.keys()))
    if not common:
        raise RuntimeError("No common IDs between pred and gold.")

    # Choose evaluation mode
    if mode == "relaxed":
        counts = evaluate_relaxed(pred, gold, common, threshold)
        mode_label = f"Relaxed Match (threshold={threshold})"
    else:
        counts = evaluate_exact(pred, gold, common)
        mode_label = "Exact Match"

    lines = ["tag,tp,fp,fn,precision,recall,f1"]
    
    # Collect per-tag F1 scores for Macro F1
    tag_f1_scores = []
    # Collect totals for Micro F1
    total_tp = total_fp = total_fn = 0
    
    for tag in TAGS:
        tp, fp, fn = counts[tag]["tp"], counts[tag]["fp"], counts[tag]["fn"]
        p, r, f1 = prf(tp, fp, fn)
        tag_f1_scores.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        lines.append(f"{tag},{tp},{fp},{fn},{p:.4f},{r:.4f},{f1:.4f}")

    # Macro F1: average of per-tag F1 scores
    macro_f1 = sum(tag_f1_scores) / len(tag_f1_scores) if tag_f1_scores else 0.0
    
    # Micro F1: computed from aggregated TP/FP/FN (PRIMARY METRIC)
    micro_p, micro_r, micro_f1 = prf(total_tp, total_fp, total_fn)
    
    lines.append(f"")
    lines.append(f"# OVERALL METRICS ({mode_label})")
    lines.append(f"MICRO_F1,{total_tp},{total_fp},{total_fn},{micro_p:.4f},{micro_r:.4f},{micro_f1:.4f}")
    lines.append(f"MACRO_F1,,,,,,{macro_f1:.4f}")
    lines.append(f"")
    lines.append(f"# PRIMARY METRIC: Micro F1 = {micro_f1:.4f}")
    lines.append(f"# SECONDARY METRIC: Macro F1 = {macro_f1:.4f}")

    Path(out_csv).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"\n{'=' * 50}")
    print(f"EVALUATION RESULTS ({mode_label})")
    print(f"{'=' * 50}")
    print(f"Common requirements: {len(common)}")
    print(f"Micro F1 (PRIMARY):   {micro_f1:.4f}")
    print(f"Macro F1 (SECONDARY): {macro_f1:.4f}")
    print(f"Precision:            {micro_p:.4f}")
    print(f"Recall:               {micro_r:.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate span predictions against gold annotations.")
    ap.add_argument("--pred", required=True, help="Path to predictions JSON")
    ap.add_argument("--gold", required=True, help="Path to gold annotations JSON")
    ap.add_argument("--out", default="results_f1.csv", help="Output CSV path")
    ap.add_argument("--mode", choices=["exact", "relaxed"], default="relaxed",
                    help="Matching mode: exact or relaxed (default: relaxed)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="IoU/TextSimilarity threshold for relaxed matching (default: 0.5)")
    args = ap.parse_args()
    main(args.pred, args.gold, args.out, mode=args.mode, threshold=args.threshold)
