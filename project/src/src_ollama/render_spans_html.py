import json
from pathlib import Path
from html import escape

TAG_STYLE = {
    "Main_actor": "background:#c7f9cc;",
    "Entity": "background:#a0c4ff;",
    "Action": "background:#ffd6a5;",
    "System_response": "background:#bdb2ff;",
    "Condition": "background:#fdffb6;",
    "Precondition": "background:#9bf6ff;",
    "Constraint": "background:#ffadad;",
    "Exception": "background:#ffc6ff;",
}

def render_one(text: str, spans):
    spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
    out = []
    cursor = 0
    for s in spans:
        out.append(escape(text[cursor:s["start"]]))
        style = TAG_STYLE.get(s["tag"], "background:#eee;")
        out.append(f"<mark style='{style} padding:2px 4px; border-radius:6px;' title='{escape(s['tag'])}'>"
                   f"{escape(text[s['start']:s['end']])}</mark>")
        cursor = s["end"]
    out.append(escape(text[cursor:]))
    return "".join(out)

def main(pred_json: str, out_html: str):
    items = json.loads(Path(pred_json).read_text(encoding="utf-8"))
    parts = ["<html><body style='font-family:Arial, sans-serif; line-height:1.6;'>"]
    for it in items:
        parts.append(f"<h3>Req {it['id']}</h3>")
        parts.append(f"<p>{render_one(it['text'], it['spans'])}</p>")
        # legend
        legend = " ".join([f"<mark style='{TAG_STYLE[t]} padding:2px 4px; border-radius:6px;'>{t}</mark>"
                           for t in TAG_STYLE])
        parts.append(f"<p style='font-size:12px;'>Legend: {legend}</p><hr/>")
    parts.append("</body></html>")
    Path(out_html).write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_html}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.pred, args.out)
