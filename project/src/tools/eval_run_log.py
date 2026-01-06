import argparse
import json
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_items(path: Path) -> List[Dict[str, Any]]:
    raw = _load_json(path)

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

    raise ValueError(f"Unsupported JSON format in {path} (expected list or dict with items/data).")


def index_by_id(items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for it in items:
        if "id" not in it:
            continue
        try:
            rid = int(it["id"])
        except Exception:
            continue
        out[rid] = it
    return out


def _to_spans(item: Dict[str, Any]) -> List[Span]:
    txt = item.get("text") or ""
    spans = item.get("spans") or []
    out: List[Span] = []
    if not isinstance(txt, str) or not isinstance(spans, list):
        return out
    for sp in spans:
        if not isinstance(sp, dict):
            continue
        tag = sp.get("tag")
        st = sp.get("start")
        en = sp.get("end")
        sp_txt = sp.get("text")
        if tag not in TAGS or not isinstance(st, int) or not isinstance(en, int) or not isinstance(sp_txt, str):
            continue
        # best effort: clamp to bounds, but keep original text field for remap
        st2 = max(0, min(st, len(txt)))
        en2 = max(st2, min(en, len(txt)))
        out.append(Span(tag=tag, start=st2, end=en2, text=sp_txt))
    return out


def iou(a: Span, b: Span) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    if inter <= 0:
        return 0.0
    la = max(0, a.end - a.start)
    lb = max(0, b.end - b.start)
    union = la + lb - inter
    return (inter / union) if union > 0 else 0.0


def text_sim(a: Span, b: Span) -> float:
    return SequenceMatcher(None, a.text, b.text).ratio()


def span_score(a: Span, b: Span, mode: str) -> float:
    if a.tag != b.tag:
        return 0.0
    if mode == "Exact":
        return 1.0 if (a.start == b.start and a.end == b.end) else 0.0
    s_iou = iou(a, b)
    if s_iou <= 0:
        return 0.0
    s_txt = text_sim(a, b)
    return 0.65 * s_iou + 0.35 * s_txt


def match_counts(pred: List[Span], gold: List[Span], mode: str, threshold: float) -> Tuple[int, int, int]:
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
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def evaluate_pred_vs_gold(
    pred_items: List[Dict[str, Any]],
    gold_map: Dict[int, Dict[str, Any]],
    *,
    mode: str,
    threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    pred_map = index_by_id(pred_items)
    common_ids = sorted(set(pred_map.keys()) & set(gold_map.keys()))

    overall_tp = overall_fp = overall_fn = 0
    rows: List[Dict[str, Any]] = []
    for tag in TAGS:
        tp_t = fp_t = fn_t = 0
        for rid in common_ids:
            p_item = pred_map.get(rid)
            g_item = gold_map.get(rid)
            if not p_item or not g_item:
                continue
            pred_spans = [s for s in _to_spans(p_item) if s.tag == tag]
            gold_spans = [s for s in _to_spans(g_item) if s.tag == tag]
            tp, fp, fn = match_counts(pred_spans, gold_spans, mode=mode, threshold=threshold)
            tp_t += tp
            fp_t += fp
            fn_t += fn
        p, r, f1 = prf(tp_t, fp_t, fn_t)
        rows.append(
            {
                "tag": tag,
                "TP": tp_t,
                "FP": fp_t,
                "FN": fn_t,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }
        )
        overall_tp += tp_t
        overall_fp += fp_t
        overall_fn += fn_t

    p_all, r_all, f1_all = prf(overall_tp, overall_fp, overall_fn)
    summary = {
        "common_requirements": len(common_ids),
        "mode": mode,
        "threshold": threshold,
        "TP": overall_tp,
        "FP": overall_fp,
        "FN": overall_fn,
        "precision": round(p_all, 4),
        "recall": round(r_all, 4),
        "f1": round(f1_all, 4),
    }
    return summary, rows


def _find_all(haystack: str, needle: str) -> List[int]:
    if not needle:
        return []
    out = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        out.append(idx)
        start = idx + 1
    return out


def _best_occurrence_by_position(occurrences: List[int], *, needle_len: int, pred_pos: float, gold_len: int) -> int:
    # pick occurrence whose normalized position is closest to predicted normalized position
    best_i = occurrences[0]
    best_d = float("inf")
    for i in occurrences:
        gpos = (i / gold_len) if gold_len else 0.0
        d = abs(gpos - pred_pos)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _fuzzy_locate(
    gold_text: str,
    target: str,
    *,
    pred_pos: float,
    window: int = 250,
    len_slack: int = 8,
) -> Tuple[Optional[int], Optional[int], float]:
    """
    Find best-matching substring in gold_text for target using SequenceMatcher ratio,
    searching near pred_pos (normalized in [0,1]) to keep it fast.
    """
    if not gold_text or not target:
        return None, None, 0.0

    glen = len(gold_text)
    tlen = len(target)
    center = int(pred_pos * glen) if glen else 0
    lo = max(0, center - window)
    hi = min(glen, center + window)

    best = (None, None, 0.0)
    for L in range(max(1, tlen - len_slack), tlen + len_slack + 1):
        if lo + L > hi:
            continue
        for st in range(lo, hi - L + 1):
            cand = gold_text[st : st + L]
            sc = SequenceMatcher(None, target, cand).ratio()
            if sc > best[2]:
                best = (st, st + L, sc)
    return best


def remap_pred_items_to_gold_text(
    pred_items: List[Dict[str, Any]],
    gold_map: Dict[int, Dict[str, Any]],
    *,
    min_fuzzy_ratio: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Remap predicted spans onto gold text by locating each pred span's text inside gold text
    (exact match preferred, else fuzzy local search). This makes evaluation possible when
    pred item 'text' differs from gold item 'text'.
    """
    pred_map = index_by_id(pred_items)
    common_ids = sorted(set(pred_map.keys()) & set(gold_map.keys()))

    total_spans = 0
    remapped_spans = 0
    dropped_spans = 0

    out_items: List[Dict[str, Any]] = []
    for rid in common_ids:
        p_item = pred_map[rid]
        g_item = gold_map[rid]
        p_text = p_item.get("text") if isinstance(p_item.get("text"), str) else ""
        g_text = g_item.get("text") if isinstance(g_item.get("text"), str) else ""

        new_item = dict(p_item)
        new_item["text"] = g_text

        p_spans = p_item.get("spans") or []
        if not isinstance(p_spans, list):
            p_spans = []

        new_spans: List[Dict[str, Any]] = []
        for sp in p_spans:
            if not isinstance(sp, dict):
                continue
            tag = sp.get("tag")
            st = sp.get("start")
            en = sp.get("end")
            sp_text = sp.get("text")
            if tag not in TAGS or not isinstance(sp_text, str) or not isinstance(st, int) or not isinstance(en, int):
                continue

            total_spans += 1
            pred_pos = (st / len(p_text)) if isinstance(p_text, str) and len(p_text) else 0.0

            occ = _find_all(g_text, sp_text)
            if occ:
                start = occ[0] if len(occ) == 1 else _best_occurrence_by_position(occ, needle_len=len(sp_text), pred_pos=pred_pos, gold_len=len(g_text))
                end = start + len(sp_text)
                new_spans.append({"tag": tag, "start": start, "end": end, "text": g_text[start:end]})
                remapped_spans += 1
                continue

            # fallback: fuzzy locate near predicted position
            fst, fen, score = _fuzzy_locate(g_text, sp_text, pred_pos=pred_pos)
            if fst is not None and fen is not None and score >= min_fuzzy_ratio:
                new_spans.append({"tag": tag, "start": fst, "end": fen, "text": g_text[fst:fen]})
                remapped_spans += 1
            else:
                dropped_spans += 1

        new_item["spans"] = new_spans
        out_items.append(new_item)

    stats = {
        "common_requirements": len(common_ids),
        "total_pred_spans": total_spans,
        "remapped_spans": remapped_spans,
        "dropped_spans": dropped_spans,
        "min_fuzzy_ratio": min_fuzzy_ratio,
    }
    return out_items, stats


def _parse_run_log_for_json_paths(run_log: Path) -> List[Path]:
    paths: List[Path] = []
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        # typical line: "Wrote 50 items to D:\...\pipeline_spans.json"
        if "Wrote" in line and "items to" in line and line.strip().endswith(".json"):
            try:
                p = line.split("items to", 1)[1].strip()
                paths.append(Path(p))
            except Exception:
                continue
    return paths


def _discover_pred_jsons(run_dir: Path) -> List[Path]:
    if not run_dir.exists():
        return []
    # typical layout:
    # run_dir/baseline/<variant>/baseline_spans.json
    # run_dir/pipeline/<variant>/pipeline_spans.json
    out = []
    out.extend(sorted(run_dir.rglob("baseline_spans.json")))
    out.extend(sorted(run_dir.rglob("pipeline_spans.json")))
    # also accept *_spans.json generally
    out.extend([p for p in sorted(run_dir.rglob("*_spans.json")) if p not in out])
    # de-dup preserving order
    seen = set()
    uniq: List[Path] = []
    for p in out:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    cols = ["tag", "TP", "FP", "FN", "precision", "recall", "f1"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    out_csv.write_text("\n".join(lines), encoding="utf-8")


def _pct_text_match(pred_items: List[Dict[str, Any]], gold_map: Dict[int, Dict[str, Any]]) -> Tuple[int, int]:
    pred_map = index_by_id(pred_items)
    common = sorted(set(pred_map.keys()) & set(gold_map.keys()))
    same = 0
    for rid in common:
        ptxt = pred_map[rid].get("text")
        gtxt = gold_map[rid].get("text")
        if isinstance(ptxt, str) and isinstance(gtxt, str) and ptxt == gtxt:
            same += 1
    return same, len(common)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a ReqFlow run (run.log or run dir) against gold.json.")
    ap.add_argument("--gold", default=str(Path("project/data/ gold.json")), help="Path to gold.json")
    ap.add_argument("--run-log", default="", help="Path to run.log (preferred if available)")
    ap.add_argument("--run-dir", default="", help="Path to run output directory (alternative)")
    ap.add_argument("--mode", default="Soft", choices=["Exact", "Soft"], help="Span matching mode")
    ap.add_argument("--threshold", type=float, default=0.5, help="Match threshold (ignored for Exact)")
    ap.add_argument("--remap", action="store_true", help="Remap pred spans onto gold text when texts differ")
    ap.add_argument("--min-fuzzy-ratio", type=float, default=0.87, help="Min fuzzy ratio to accept a remapped span")
    ap.add_argument("--out-dir", default="", help="Directory to write results (default: alongside pred json)")
    args = ap.parse_args(argv)

    gold_path = Path(args.gold)
    gold_items = load_items(gold_path)
    gold_map = index_by_id(gold_items)

    pred_paths: List[Path] = []
    if args.run_log:
        rl = Path(args.run_log)
        pred_paths.extend(_parse_run_log_for_json_paths(rl))
        # if log parsing fails, fall back to run-dir = log's parent
        if not pred_paths:
            pred_paths.extend(_discover_pred_jsons(rl.parent))
    elif args.run_dir:
        pred_paths.extend(_discover_pred_jsons(Path(args.run_dir)))
    else:
        print("ERROR: Provide --run-log or --run-dir", file=sys.stderr)
        return 2

    pred_paths = [p for p in pred_paths if p.name.endswith(".json") and p.exists()]
    if not pred_paths:
        print("ERROR: No prediction JSONs found from the given run.", file=sys.stderr)
        return 2

    exit_code = 0
    for pred_path in pred_paths:
        pred_items = load_items(pred_path)
        same, total = _pct_text_match(pred_items, gold_map)
        text_match_ratio = (same / total) if total else 0.0

        effective_items = pred_items
        remap_stats = None
        if text_match_ratio < 1.0 and args.remap:
            effective_items, remap_stats = remap_pred_items_to_gold_text(
                pred_items, gold_map, min_fuzzy_ratio=float(args.min_fuzzy_ratio)
            )

        summary, rows = evaluate_pred_vs_gold(
            effective_items,
            gold_map,
            mode=args.mode,
            threshold=float(args.threshold if args.mode != "Exact" else 1.0),
        )

        summary["pred_path"] = str(pred_path)
        summary["gold_path"] = str(gold_path)
        summary["text_match"] = {"same": same, "total": total, "ratio": round(text_match_ratio, 4)}
        summary["remap"] = remap_stats or {"enabled": False}

        out_base_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
        out_base_dir.mkdir(parents=True, exist_ok=True)
        stem = pred_path.stem
        out_json = out_base_dir / f"{stem}.eval.json"
        out_csv = out_base_dir / f"{stem}.eval.csv"

        out_json.write_text(json.dumps({"summary": summary, "by_tag": rows}, indent=2), encoding="utf-8")
        _write_csv(rows, out_csv)

        print(f"[eval] pred={pred_path}")
        print(f"[eval] text_match={same}/{total} ({text_match_ratio:.2%}) remap={'on' if args.remap else 'off'}")
        print(f"[eval] f1={summary['f1']} precision={summary['precision']} recall={summary['recall']}")
        print(f"[eval] wrote: {out_csv}")
        print(f"[eval] wrote: {out_json}")

        # If texts don't match and remap is disabled, results are likely meaningless.
        if text_match_ratio < 1.0 and not args.remap:
            exit_code = 3

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


