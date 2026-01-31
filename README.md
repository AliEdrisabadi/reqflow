# ReqFlow 

<img src="photo.jpg" width="300" alt="explain">




This project is a small student tool for the LLM4SE **A3** assignment: **Analysing Software Requirements Through Abstractions**.

It extracts *abstraction spans* from requirements using a local LLM via **Ollama**, renders them as **HTML highlights**, and evaluates predictions against a **gold** annotation.

## Supported modes

### 1) Baseline (single agent)
One LLM call per requirement. The model extracts **all tags at once**.

### 2) Pipeline (multi‑agent agent‑chain)
A workflow that follows the course slide idea (multi‑step / multi‑agent):

1. **Segmenter Agent** (pre‑processing): splits the requirement into clause‑like segments.
2. **Entities Agent**: `Entity`, `Main_actor`
3. **Actions Agent**: `Action`, `System_response`
4. **Logic Agent**: `Condition`, `Precondition`, `Trigger`
5. **Purpose Agent**: `Purpose`



## Tags (A3 slide taxonomy)
`Purpose, Trigger, Precondition, Condition, Action, System_response, Entity, Main_actor`

## Evaluation policy (important)
For the main comparison **we ignore character offsets** and evaluate only the extracted **(tag, span_text)** pairs.
Offsets (`start/end`) are computed only to render HTML highlights.

This makes the baseline vs multi‑agent comparison focus on **tagging / extraction quality**, not on character index counting.

---

## Setup

### Requirements
- Python 3.10+
- Ollama installed and running (default `http://localhost:11434`)

### Install
```bash
pip install -r requirements.txt
```

### Configure environment
Copy `.env.example` to `.env` and edit if needed:
```bash
cp .env.example .env
```

Key settings:
- `OLLAMA_MODEL` (e.g., `qwen3:4b-instruct`, `llama3.1:8b-instruct`, etc.)
- prompt variants:
  - `REQFLOW_DEFAULT_BASELINE_VARIANT=zero|one|few`
  - `REQFLOW_DEFAULT_PIPELINE_VARIANT=zero|one|few`

---

## Run from CLI

### Baseline
```bash
python cli.py baseline --variant one --ids 1,2,3,4,5
```

### Pipeline (multi‑agent)
```bash
python cli.py pipeline --variant one --ids 1,2,3,4,5
```

Each run writes:
- a JSON prediction file
- an HTML visualization file (same name, `.html`)

### Evaluate (tag + span_text only)
```bash
python evaluate.py --pred results/run_*/pipeline_one.json --gold data/gold.json --out results_eval.csv
```

---

## Run GUI (optional)
```bash
python reqflow.py
```

---

## Data format

### Dataset CSV (`data/dataset.csv`)
Columns:
- `id` (int)
- `text` (string)
Other columns (like `nested`, `type`, etc.) are allowed and ignored.

### Gold JSON (`data/gold.json`)
List/dict format is supported; each item has:
- `id`
- `text`
- `spans`: objects like `{ "tag": "...", "text": "...", "start": ..., "end": ... }`

### Prediction JSON
Same structure as gold, but `start/end` are optional and only used for HTML rendering.

---

## Notes for best results
- Prefer `one` or `few` variants for better copy‑exact behavior.
- If the model starts paraphrasing, increase strictness in prompts (already done in provided prompts).
