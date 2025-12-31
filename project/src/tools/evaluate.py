import argparse
import json
from collections import defaultdict
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


def to_set(spans: Any):
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


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
    return p, r, f1


def main(pred_path: str, gold_path: str, out_csv: str):
    pred = index_by_id(load_items(pred_path))
    gold = index_by_id(load_items(gold_path))

    common = sorted(set(pred.keys()) & set(gold.keys()))
    if not common:
        raise RuntimeError("No common IDs between pred and gold.")

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

    lines = ["tag,tp,fp,fn,precision,recall,f1"]
    macro = 0.0
    for tag in TAGS:
        tp, fp, fn = counts[tag]["tp"], counts[tag]["fp"], counts[tag]["fn"]
        p, r, f1 = prf(tp, fp, fn)
        macro += f1
        lines.append(f"{tag},{tp},{fp},{fn},{p:.4f},{r:.4f},{f1:.4f}")

    macro /= len(TAGS)
    lines.append(f"MACRO_AVG,,,,,,{macro:.4f}")

    Path(out_csv).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default="results_f1.csv")
    args = ap.parse_args()
    main(args.pred, args.gold, args.out)
