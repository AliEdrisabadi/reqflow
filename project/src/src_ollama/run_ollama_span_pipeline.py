import json
import pandas as pd
from pathlib import Path
from ollama_client import ollama_generate

def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def fill(template: str, **kwargs) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{{" + k + "}}", v)
    return out

def validate_clauses(text: str, clauses):
    clauses = sorted(clauses, key=lambda c: c["start"])
    cursor = 0
    for c in clauses:
        assert c["start"] == cursor, "Gap or misaligned clause start"
        assert text[c["start"]:c["end"]] == c["text"], "Clause text mismatch"
        assert c["end"] >= c["start"]
        cursor = c["end"]
    assert cursor == len(text), "Clauses do not cover full text"
    return clauses

def validate_spans(text: str, spans):
    spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
    last_end = -1
    for s in spans:
        assert 0 <= s["start"] <= s["end"] <= len(text)
        assert s["start"] >= last_end, "Overlapping spans"
        assert text[s["start"]:s["end"]] == s["text"], "Span text mismatch"
        last_end = s["end"]
    return spans

def main(dataset_csv: str, out_json: str, model: str, ids: str = ""):
    df = pd.read_csv(dataset_csv)
    if ids.strip():
        wanted = set(int(x) for x in ids.split(","))
        df = df[df["id"].isin(wanted)].copy()

    p_segment = load_prompt("../prompts_ollama/span_step1_segment.md")
    p_tag = load_prompt("../prompts_ollama/span_step2_tag.md")

    outputs = []
    for _, row in df.iterrows():
        rid = int(row["id"])
        text = str(row["text_en"])

        seg_prompt = fill(p_segment, REQUIREMENT_TEXT=text)
        seg_out = ollama_generate(seg_prompt, model=model)
        clauses = validate_clauses(text, seg_out.get("clauses", []))

        tag_prompt = fill(p_tag, REQUIREMENT_TEXT=text, CLAUSES_JSON=json.dumps({"clauses": clauses}))
        tag_out = ollama_generate(tag_prompt, model=model)
        spans = validate_spans(text, tag_out.get("spans", []))

        outputs.append({"id": rid, "text": text, "clauses": clauses, "spans": spans})

    Path(out_json).write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} items to {out_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="Override model (else uses OLLAMA_MODEL from .env)")
    ap.add_argument("--ids", default="", help="Comma-separated requirement IDs (optional)")
    args = ap.parse_args()
    main(args.dataset, args.out, args.model, args.ids)
