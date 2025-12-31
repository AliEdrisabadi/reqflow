# ReqFlow


![ReqFlow screenshot](/img.png)



ReqFlow is a small tool to run **span annotation** on requirements using a **local LLM via Ollama**, render results to **HTML**, and optionally compute a basic **evaluation** (Precision/Recall/F1) against a gold JSON.

It supports two modes:
- **Baseline**: one-shot span tagging (single-step)
- **Pipeline**: segment → tag (multi-step)

It also includes a **PyQt6 GUI** to:
- select a dataset CSV
- choose requirement IDs (checkbox list)
- run Baseline / Pipeline / Both
- view the HTML output
- run evaluation (per-tag P/R/F1 + threshold curve)

---

## Expected project structure

This README matches the folder names you currently use: `src/` and `prompts/`.

```
project/
  reqflow.py
  .env                      (optional)
  requirements.txt          (recommended)
  data/
    requirements_dataset.csv
    gold_annotations_seat_based.json
  prompts/
    baseline.md
    segment.md
    tag.md
    span_rewrite.md         (optional)
  src/
    baseline.py
    pipeline.py
    render.py
    cli.py
    env.py
    ollama.py
    tools/
      evaluate.py
  result_gui/
  result_cli/
```

---

## Requirements

- **Python 3.12**
- **Ollama** installed and running  
  - default host: `http://localhost:11434`

---

## 1) Create a venv (Python 3.12) and install dependencies (no conda)

### Windows (PowerShell)
```bash
cd <PROJECT_ROOT>
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (CMD)
```bash
cd <PROJECT_ROOT>
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS
```bash
cd <PROJECT_ROOT>
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) Ollama: run + pull a model

Make sure Ollama is running, then pull a model:

```bash
ollama pull qwen3:4b-instruct
```

You may use any instruct-capable model that reliably outputs JSON.

---

## 3) Optional: configure `.env`

Create a file named `.env` in the project root (recommended):

```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct
OLLAMA_TIMEOUT=120
OLLAMA_TEMPERATURE=0.2
OLLAMA_NUM_PREDICT=512
```

These keys are read by `src/ollama.py` (see `get_config()`).

---

## 4) Run the GUI

The GUI entry point is `reqflow.py` (project root):

```bash
python reqflow.py
```

### Typical GUI workflow
1. Select the dataset CSV (must include `id` and `text_en`)
2. Tick the requirement IDs you want to analyze
3. Choose method: **Baseline / Pipeline / Both**
4. Click **Analysis**
5. The generated HTML is shown in the **Main** tab
6. In the **Evaluation** tab:
   - select a gold JSON file
   - choose mode (Exact / Relaxed) + threshold
   - click **Compute P/R/F1**

GUI runs are saved under:
- `result_gui/run_<timestamp>/`

---

## 5) Run from the command line (CLI)

CLI runner is `src/cli.py`. Run from the project root:

### Run both (baseline + pipeline)
```bash
python src/cli.py --mode both --dataset data/requirements_dataset.csv --ids 1,2,3
```

### Baseline only
```bash
python src/cli.py --mode baseline --dataset data/requirements_dataset.csv --ids 1,2,3
```

### Pipeline only
```bash
python src/cli.py --mode pipeline --dataset data/requirements_dataset.csv --ids 1,2,3
```

CLI runs are saved under:
- `result_cli/run_<timestamp>/`

Each run typically generates:
- `baseline_spans.json` + `baseline_spans.html` (if baseline enabled)
- `pipeline_spans.json` + `pipeline_spans.html` (if pipeline enabled)

---

## 6) Evaluation (CLI)

There is also a standalone evaluation script:

- `src/tools/evaluate.py`

Example:

```bash
python src/tools/evaluate.py \
  --pred result_cli/run_YYYYMMDD_HHMMSS/pipeline_spans.json \
  --gold data/gold_annotations_seat_based.json \
  --out results_f1.csv
```

Output: a CSV with per-tag metrics and macro-average.

---

## 7) Data formats

### Dataset CSV
Expected columns:
- `id` (int)
- `text_en` (string)

Example:
```csv
id,text_en
1,"The system shall ..."
2,"The user can ..."
```

### Prediction JSON
A list of items like:
- `id`
- `text`
- `spans` (each span has: `tag`, `start`, `end`, `text`)

### Gold JSON
GUI supports multiple gold formats, including:
- list of `{id, spans}`
- dict with `items` or `data` arrays
- dict keyed by id

---

## 8) Common issues

### Ollama not reachable
- Ensure Ollama is running
- Verify `OLLAMA_HOST` (default: `http://localhost:11434`)

### Model not found
```bash
ollama pull <model-name>
```

### GUI HTML view not working
Install WebEngine:
```bash
pip install PyQt6-WebEngine
```

### CSV column error
Your dataset must contain `id` and `text_en`.

---

## Quick start (minimal)

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen3:4b-instruct
python reqflow.py
```
