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

def validate_spans(text: str, spans):

    spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
    last_end = -1
    for s in spans:
        assert 0 <= s["start"] <= s["end"] <= len(text)
        assert s["start"] >= last_end, "Overlapping spans"
        assert text[s["start"]:s["end"]] == s["text"], "Text mismatch"
        last_end = s["end"]
    return spans

def main(
    dataset_csv: str,
    out_json: str,
    model: str,
    prompt_path: str,
    ids: str = "",
):
    df = pd.read_csv(dataset_csv)
    if ids.strip():
        wanted = set(int(x) for x in ids.split(","))
        df = df[df["id"].isin(wanted)].copy()

    tmpl = load_prompt(prompt_path)
    outputs = []
    for _, row in df.iterrows():
        rid = int(row["id"])
        text = str(row["text_en"])
        prompt = fill(tmpl, REQUIREMENT_TEXT=text)
        out = ollama_generate(prompt, model=model)
        spans = out.get("spans", [])
        spans = validate_spans(text, spans)
        outputs.append({"id": rid, "text": text, "spans": spans})

    Path(out_json).write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(outputs)} items to {out_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="Override model (else uses OLLAMA_MODEL from .env)")
    ap.add_argument("--prompt", default="../prompts_ollama/span_baseline.md")
    ap.add_argument("--ids", default="", help="Comma-separated requirement IDs (optional)")
    args = ap.parse_args()
    main(args.dataset, args.out, args.model, args.prompt, args.ids)
