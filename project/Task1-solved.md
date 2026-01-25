# ReqFlow - Project Overview

## High-Level Description

**ReqFlow** is a tool for performing **span annotation** on software requirements using a **local LLM via Ollama**. It extracts and tags semantic spans from requirement texts, renders the results to HTML visualizations, and computes evaluation metrics (Precision/Recall/F1) against gold standard annotations.

### Key Features
- **Two Processing Modes:**
  - **Baseline**: Single-step span tagging (direct annotation)
  - **Pipeline**: Multi-step approach (segment → tag)
- **GUI Application** (PyQt6) for interactive analysis
- **CLI Interface** for batch processing
- **HTML Visualization** of annotated spans
- **Evaluation System** with P/R/F1 metrics and threshold curves

### Supported Tags
The system annotates requirements with 8 semantic tags:
1. `Main_actor` - Primary actor in the requirement
2. `Entity` - Objects/entities mentioned
3. `Action` - Actions to be performed
4. `System_response` - System responses/outputs
5. `Condition` - Conditional statements
6. `Precondition` - Prerequisites
7. `Constraint` - Constraints/limitations
8. `Exception` - Exception handling cases

---

## Project Structure

```
reqflow/
├── project/
│   ├── reqflow.py              # Main GUI application (entry point)
│   ├── .env                    # Environment configuration
│   ├── requirements.txt        # Python dependencies
│   ├── data/
│   │   ├── dataset.csv         # Input requirements dataset
│   │   └── gold.json           # Gold standard annotations
│   ├── prompts/                # LLM prompt templates
│   │   ├── baseline_*.md       # Baseline prompts (zero/one/few shot)
│   │   ├── segment_*.md        # Segmentation prompts
│   │   ├── tag_*.md            # Tagging prompts
│   │   └── span_rewrite.md     # Span rewriting prompt
│   └── src/
│       ├── cli.py              # CLI entry point
│       ├── env.py              # Environment/dotenv loader
│       ├── ollama.py           # Ollama API client
│       ├── baseline.py         # Baseline annotation logic
│       ├── pipeline.py         # Pipeline annotation logic
│       ├── render.py           # HTML renderer
│       └── tools/
│           └── evaluate.py     # Evaluation metrics
├── report/                     # LaTeX report files
├── slides/                     # Presentation slides
└── README.md                   # Project documentation
```

---

## Model Architecture

This project uses **Decoder-Only Transformer LLMs** via Ollama (not encoder-decoder):

### Current Configuration (from `.env`):
```
OLLAMA_MODEL=qwen3:4b-instruct
```

### Default Fallback (in code):
```
llama3.1:8b
```

| Model | Type | Architecture |
|-------|------|--------------|
| **Qwen3:4b-instruct** | Decoder-only | Causal Language Model |
| **LLaMA 3.1:8b** | Decoder-only | Causal Language Model |

The project uses a **prompt-based approach** instead of traditional NER/sequence labeling:

| Traditional NER | This Project (ReqFlow) |
|-----------------|------------------------|
| BERT Encoder + CRF | Decoder-only LLM |
| Token-level classification | JSON generation |
| Fine-tuned model | Zero/One/Few-shot prompting |
| Requires training | No training needed |

---

## Python File Order Flow

### Entry Points

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                             │
│                                                                      │
│    ┌──────────────┐                      ┌──────────────┐           │
│    │  reqflow.py  │  (GUI)               │   cli.py     │  (CLI)    │
│    │  Entry Point │                      │  Entry Point │           │
│    └──────┬───────┘                      └──────┬───────┘           │
└───────────┼──────────────────────────────────────┼──────────────────┘
            │                                      │
            └──────────────┬───────────────────────┘
                           ▼
```

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     1. INITIALIZATION LAYER                          │
│                                                                      │
│    ┌──────────────────────────────────────────────────────────┐     │
│    │                      env.py                               │     │
│    │  • find_dotenv() - Locate .env file                      │     │
│    │  • load_dotenv() - Load environment variables            │     │
│    └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     2. LLM CLIENT LAYER                              │
│                                                                      │
│    ┌──────────────────────────────────────────────────────────┐     │
│    │                     ollama.py                             │     │
│    │  • OllamaConfig - Configuration dataclass                │     │
│    │  • get_config() - Load Ollama settings from env          │     │
│    │  • ollama_check() - Verify Ollama connectivity           │     │
│    │  • ollama_generate() - Send prompts, get JSON responses  │     │
│    │  • _extract_json() - Parse JSON from LLM output          │     │
│    └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     3. PROCESSING LAYER                              │
│                                                                      │
│  ┌───────────────────────────┐     ┌───────────────────────────┐    │
│  │      baseline.py          │     │      pipeline.py          │    │
│  │   (1 LLM Call/req)        │     │   (2 LLM Calls/req)       │    │
│  ├───────────────────────────┤     ├───────────────────────────┤    │
│  │                           │     │                           │    │
│  │ 1. Load baseline prompt   │     │ 1. Load segment prompt    │    │
│  │                           │     │ 2. Load tag prompt        │    │
│  │ 2. Fill {{REQUIREMENT}}   │     │                           │    │
│  │                           │     │ 3. Fill segment template  │    │
│  │ 3. Call Ollama            │     │    {{REQUIREMENT_TEXT}}   │    │
│  │    → Get spans directly   │     │                           │    │
│  │                           │     │ 4. Call Ollama #1         │    │
│  │ 4. Validate spans         │     │    → Get clauses          │    │
│  │    • Check offsets        │     │                           │    │
│  │    • Repair if needed     │     │ 5. Validate clauses       │    │
│  │    • Deduplicate          │     │    • Check offsets        │    │
│  │                           │     │    • Fallback to full text│    │
│  │ 5. Output JSON            │     │                           │    │
│  │                           │     │ 6. Fill tag template      │    │
│  │                           │     │    {{REQUIREMENT_TEXT}}   │    │
│  │                           │     │    {{CLAUSES_JSON}}       │    │
│  │                           │     │                           │    │
│  │                           │     │ 7. Call Ollama #2         │    │
│  │                           │     │    → Get spans            │    │
│  │                           │     │                           │    │
│  │                           │     │ 8. Validate spans         │    │
│  │                           │     │                           │    │
│  │                           │     │ 9. Output JSON            │    │
│  │                           │     │    (with clauses+spans)   │    │
│  └───────────────────────────┘     └───────────────────────────┘    │
│                                                                      │
│    Shared Validation Logic:                                         │
│    ┌──────────────────────────────────────────────────────────┐     │
│    │ validate_spans():                                        │     │
│    │   • Check tag ∈ {Main_actor, Entity, Action, ...}       │     │
│    │   • Verify 0 ≤ start ≤ end ≤ len(text)                  │     │
│    │   • Verify text[start:end] == span.text                 │     │
│    │   • If invalid → _repair_span() using text search       │     │
│    │   • Deduplicate by (tag, start, end, text)              │     │
│    │   • Sort by position                                    │     │
│    └──────────────────────────────────────────────────────────┘     │
│                                                                      │
│    _repair_span():                                                  │
│    ┌──────────────────────────────────────────────────────────┐     │
│    │   • Normalize quotes: "" → "", '' → ''                  │     │
│    │   • Normalize whitespace: \u00a0 → regular space        │     │
│    │   • Find all occurrences in text using regex            │     │
│    │   • Pick closest match to original start position       │     │
│    └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     4. OUTPUT LAYER                                  │
│                                                                      │
│    ┌──────────────────────────────────────────────────────────┐     │
│    │                      render.py                            │     │
│    │  • _render_item() - Generate HTML for single requirement │     │
│    │  • _bars_for_tag() - Create visual span bars             │     │
│    │  • main() - Generate complete HTML document              │     │
│    │  • Supports dark/light themes                            │     │
│    └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     5. EVALUATION LAYER                              │
│                                                                      │
│    ┌──────────────────────────────────────────────────────────┐     │
│    │               tools/evaluate.py                           │     │
│    │  • load_items() - Parse prediction/gold JSON             │     │
│    │  • index_by_id() - Index items by requirement ID         │     │
│    │  • to_set() - Convert spans to comparable sets           │     │
│    │  • prf() - Calculate Precision/Recall/F1                 │     │
│    │  • main() - Generate evaluation CSV                      │     │
│    └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Detailed File Descriptions

### 1. `env.py` - Environment Configuration
**Purpose:** Load environment variables from `.env` file
**Key Functions:**
- `find_dotenv(start, filename)` - Search upward from directory to find `.env`
- `load_dotenv(path, override)` - Parse and load environment variables

**Called by:** All other modules at initialization

---

### 2. `ollama.py` - LLM Client
**Purpose:** Interface with local Ollama server for LLM inference
**Key Functions:**
- `get_config()` - Build `OllamaConfig` from environment variables
- `ollama_check(timeout)` - Verify Ollama server is reachable
- `ollama_generate(prompt, model, ...)` - Send prompt and get JSON response

**Configuration via environment:**
- `OLLAMA_HOST` - Server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL` - Model name (default: `llama3.1:8b`)
- `OLLAMA_TIMEOUT` - Request timeout (default: 120s)
- `OLLAMA_TEMPERATURE` - Generation temperature (default: 0.2)
- `OLLAMA_NUM_PREDICT` - Max tokens (default: 512)

**Called by:** `baseline.py`, `pipeline.py`

---

### 3. `baseline.py` - Single-Step Annotation
**Purpose:** Direct span tagging in one LLM call (simpler but potentially less accurate)

**Key Functions:**
- `main(dataset_csv, out_json, model, prompt_path, ids)` - Main entry point
- `validate_spans(text, spans)` - Verify span integrity
- `_repair_span(text, span)` - Fix broken span offsets
- `_normalize_variants(s)` - Handle quote/whitespace variations
- `_find_occurrences(text, needle)` - Find text matches using regex

---

#### Detailed Logic Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    BASELINE PROCESSING FLOW                      │
└─────────────────────────────────────────────────────────────────┘

1. INITIALIZATION
   ┌────────────────────────────────────────────────────────────┐
   │  _resolve_prompt_path()                                    │
   │  Priority:                                                 │
   │    1) Explicit prompt_path argument                        │
   │    2) Environment: REQFLOW_BASELINE_PROMPT                 │
   │    3) Fallback: prompts/baseline.md                        │
   └────────────────────────────────────────────────────────────┘
                              │
                              ▼
2. LOAD DATA
   ┌────────────────────────────────────────────────────────────┐
   │  • Read CSV with pandas                                    │
   │  • Auto-detect text column: text_en > text > requirement   │
   │  • Filter by IDs if provided (comma-separated)             │
   └────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. FOR EACH REQUIREMENT
   ┌────────────────────────────────────────────────────────────┐
   │  a) TEMPLATE FILLING                                       │
   │     fill(template, REQUIREMENT_TEXT=text)                  │
   │     Replaces {{REQUIREMENT_TEXT}} with actual text         │
   │                                                            │
   │  b) LLM CALL                                               │
   │     ollama_generate(prompt) → JSON response                │
   │     Expected output: {"spans": [...]}                      │
   │                                                            │
   │  c) SPAN VALIDATION (see detailed logic below)             │
   │     validate_spans(text, spans) → cleaned spans            │
   └────────────────────────────────────────────────────────────┘
                              │
                              ▼
4. OUTPUT
   ┌────────────────────────────────────────────────────────────┐
   │  Write JSON: [{"id": X, "text": "...", "spans": [...]}]    │
   └────────────────────────────────────────────────────────────┘
```

---

#### Span Validation Logic (`validate_spans`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPAN VALIDATION ALGORITHM                     │
└─────────────────────────────────────────────────────────────────┘

FOR EACH span in LLM output:
│
├─► 1. TYPE CHECK
│      • Is it a dict? (skip if not)
│      • Is tag in TAGS list? (skip if not)
│
├─► 2. OFFSET VALIDATION
│      Check if span is valid:
│      • start is int
│      • end is int  
│      • 0 <= start <= end <= len(text)
│      • text[start:end] == span["text"]
│
│      ┌─────────────────────────────────────────────┐
│      │  IF VALID → Accept span as-is              │
│      │  IF INVALID → Attempt repair               │
│      └─────────────────────────────────────────────┘
│
├─► 3. SPAN REPAIR (_repair_span)
│      │
│      ├── Generate text variants:
│      │   • Original text
│      │   • Normalized quotes: "" → "", '' → ''
│      │   • Normalized whitespace: \u00a0 → space
│      │
│      ├── Find all occurrences in requirement text
│      │
│      ├── Sort occurrences:
│      │   • If original start exists → sort by distance to it
│      │   • Otherwise → sort by position (first occurrence)
│      │
│      └── Use closest/first match as corrected span
│
├─► 4. DEDUPLICATION
│      Key = (tag, start, end, text)
│      Skip if already seen
│
└─► 5. SORT OUTPUT
       Sort by (start, end, tag)
```

---

#### Example Span Repair Scenario

```
Original requirement: "The user shall be able to login."
                       0123456789...

LLM returns:
{
  "tag": "Main_actor",
  "start": 5,        ← Wrong offset!
  "end": 9,
  "text": "user"     ← Correct text
}

Repair process:
1. text[5:9] = "ser " ≠ "user" → Invalid!
2. Search for "user" in text
3. Found at position 4-8
4. Corrected span: {"tag": "Main_actor", "start": 4, "end": 8, "text": "user"}
```

**Called by:** `reqflow.py` (GUI), `cli.py` (CLI)

---

### 4. `pipeline.py` - Two-Step Annotation
**Purpose:** Segment requirements into clauses first, then tag spans (more structured, potentially more accurate)

**Key Functions:**
- `main(dataset_csv, out_json, model, ids, segment_prompt_path, tag_prompt_path)` - Main entry
- `validate_clauses(requirement_text, clauses)` - Verify clause segmentation
- `validate_spans(text, spans)` - Verify span integrity (same as baseline)
- `_resolve_pipeline_prompts()` - Resolve both prompt paths

---

#### Detailed Logic Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE PROCESSING FLOW                      │
│                      (Two LLM Calls)                             │
└─────────────────────────────────────────────────────────────────┘

1. INITIALIZATION
   ┌────────────────────────────────────────────────────────────┐
   │  _resolve_pipeline_prompts()                               │
   │                                                            │
   │  Segment Prompt Priority:                                  │
   │    1) segment_prompt_path argument                         │
   │    2) Environment: REQFLOW_SEGMENT_PROMPT                  │
   │    3) Fallback: prompts/segment.md                         │
   │                                                            │
   │  Tag Prompt Priority:                                      │
   │    1) tag_prompt_path argument                             │
   │    2) Environment: REQFLOW_TAG_PROMPT                      │
   │    3) Fallback: prompts/tag.md                             │
   └────────────────────────────────────────────────────────────┘
                              │
                              ▼
2. LOAD DATA (same as baseline)
   ┌────────────────────────────────────────────────────────────┐
   │  • Read CSV, detect text column, filter by IDs             │
   └────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. FOR EACH REQUIREMENT
   │
   │  ┌──────────────────────────────────────────────────────┐
   │  │           STEP 1: SEGMENTATION (LLM Call #1)         │
   │  └──────────────────────────────────────────────────────┘
   │  │
   │  │  a) Fill segment template:
   │  │     fill(tmpl1, REQUIREMENT_TEXT=text)
   │  │
   │  │  b) Call Ollama:
   │  │     ollama_generate(prompt1) → {"clauses": [...]}
   │  │
   │  │  c) Validate clauses:
   │  │     validate_clauses(text, clauses)
   │  │
   │  │  Expected clause structure:
   │  │  {
   │  │    "clause_id": 1,
   │  │    "start": 0,
   │  │    "end": 25,
   │  │    "text": "The user shall login",
   │  │    "cue": "MAIN"  ← Cue word (IF, WHEN, SHALL, etc.)
   │  │  }
   │  │
   │  ▼
   │  ┌──────────────────────────────────────────────────────┐
   │  │           STEP 2: TAGGING (LLM Call #2)              │
   │  └──────────────────────────────────────────────────────┘
   │  │
   │  │  a) Fill tag template with BOTH inputs:
   │  │     fill(tmpl2, 
   │  │          REQUIREMENT_TEXT=text,
   │  │          CLAUSES_JSON=json.dumps(clauses))
   │  │
   │  │  b) Call Ollama:
   │  │     ollama_generate(prompt2) → {"spans": [...]}
   │  │
   │  │  c) Validate spans:
   │  │     validate_spans(text, spans)
   │  │
   │  ▼
   └──► Output: {"id": X, "text": "...", "clauses": [...], "spans": [...]}
                              │
                              ▼
4. OUTPUT
   ┌────────────────────────────────────────────────────────────┐
   │  Write JSON with both clauses and spans                    │
   └────────────────────────────────────────────────────────────┘
```

---

#### Clause Validation Logic (`validate_clauses`)

```
┌─────────────────────────────────────────────────────────────────┐
│                   CLAUSE VALIDATION ALGORITHM                    │
└─────────────────────────────────────────────────────────────────┘

INPUT: List of clauses from LLM

├─► CASE 1: Empty or invalid input
│      Return fallback: entire text as single clause
│      [{"clause_id": 1, "start": 0, "end": len(text), 
│        "text": full_text, "cue": "MAIN"}]
│
├─► CASE 2: Valid clauses exist
│      FOR EACH clause:
│      │
│      ├── Check offset validity:
│      │   • start/end are integers
│      │   • 0 <= start <= end <= len(text)
│      │   • text[start:end] == clause["text"]
│      │
│      ├── IF VALID → Accept clause
│      │
│      ├── IF INVALID → Attempt text search repair:
│      │   • Search for clause["text"] in requirement
│      │   • Use first match position
│      │
│      └── Extract: clause_id, start, end, text, cue
│
└─► FALLBACK: If no valid clauses found
       Return entire text as single clause
```

---

#### Why Pipeline is More Structured

```
┌─────────────────────────────────────────────────────────────────┐
│                  BASELINE vs PIPELINE COMPARISON                 │
└─────────────────────────────────────────────────────────────────┘

BASELINE (Single Step):
┌─────────────────────────────────────────────────────────────────┐
│  "The system shall allow users to login when authenticated."    │
│                              │                                  │
│                              ▼                                  │
│                    [Single LLM Call]                            │
│                              │                                  │
│                              ▼                                  │
│  Direct span extraction (may miss complex structures)           │
└─────────────────────────────────────────────────────────────────┘

PIPELINE (Two Steps):
┌─────────────────────────────────────────────────────────────────┐
│  "The system shall allow users to login when authenticated."    │
│                              │                                  │
│                              ▼                                  │
│                    [LLM Call #1: Segment]                       │
│                              │                                  │
│                              ▼                                  │
│  Clauses:                                                       │
│    1. "The system shall allow users to login" (cue: SHALL)      │
│    2. "when authenticated" (cue: WHEN)                          │
│                              │                                  │
│                              ▼                                  │
│                    [LLM Call #2: Tag]                           │
│                    (with clause context)                        │
│                              │                                  │
│                              ▼                                  │
│  More accurate spans because LLM understands structure          │
└─────────────────────────────────────────────────────────────────┘
```

---

#### Cue Words in Clauses

The segmentation step identifies **cue words** that mark clause boundaries:

| Cue Type | Examples | Purpose |
|----------|----------|---------|
| MAIN | (no cue) | Primary requirement statement |
| SHALL | shall, must, will | Obligation/functionality |
| IF | if, in case | Conditional |
| WHEN | when, whenever | Temporal condition |
| UNLESS | unless, except | Exception |
| WHILE | while, during | Duration |
| AFTER/BEFORE | after, before | Sequence |

**Called by:** `reqflow.py` (GUI), `cli.py` (CLI)

---

### 5. `render.py` - HTML Visualization
**Purpose:** Generate interactive HTML visualizations of annotated spans
**Key Functions:**
- `main(pred_json, out_html, theme)` - Generate HTML file
- `_render_item(item)` - Render single requirement card
- `_bars_for_tag(text, spans)` - Generate span visualization bars

**Features:**
- Color-coded tags with hover tooltips
- Dark/light theme support
- Responsive layout

**Called by:** `reqflow.py` (GUI), `cli.py` (CLI)

---

### 6. `cli.py` - Command-Line Interface
**Purpose:** Run analysis from command line
**Key Functions:**
- `main()` - Parse arguments and run analysis
- `run(mode, dataset, out_dir, ids, baseline_variant, pipeline_variant)` - Execute analysis

**Arguments:**
- `--dataset` - Path to CSV file
- `--mode` - `baseline`, `pipeline`, or `both`
- `--ids` - Comma-separated requirement IDs
- `--outdir` - Output directory
- `--baseline_variant` / `--pipeline_variant` - Prompt variant (zero/one/few)

**Calls:** `baseline.py`, `pipeline.py`, `render.py`

---

### 7. `tools/evaluate.py` - Evaluation Metrics
**Purpose:** Compute evaluation metrics against gold annotations
**Key Functions:**
- `main(pred_path, gold_path, out_csv)` - Main evaluation entry
- `prf(tp, fp, fn)` - Calculate Precision/Recall/F1
- `to_set(spans)` - Convert spans to comparable set format

**Output:** CSV with per-tag metrics and macro-average F1

**Called by:** `reqflow.py` (GUI), standalone CLI

---

### 8. `reqflow.py` - Main GUI Application
**Purpose:** PyQt6 graphical interface for interactive analysis
**Key Components:**
- `ReqFlowApp` - Main window class
- `WorkerThread` - Background processing thread
- `MetricLineChart` - Matplotlib visualization widget

**Features:**
- Dataset browsing and requirement selection
- Method selection (Baseline/Pipeline/Both)
- Prompt variant selection (Zero/One/Few-shot)
- Real-time HTML preview
- Interactive evaluation with threshold curves

**Calls:** All other modules

---

## Execution Flow Summary

### GUI Mode (`reqflow.py`)
```
User → GUI → Select Dataset → Check Requirements → Choose Method
                                                         │
                    ┌────────────────────────────────────┼────────────────────────────────────┐
                    │                                    │                                    │
                    ▼                                    ▼                                    ▼
              [Baseline Mode]                    [Pipeline Mode]                        [Both Modes]
                    │                                    │                                    │
                    ▼                                    ▼                                    │
              baseline.py                          pipeline.py                               │
                    │                                    │                                    │
                    └────────────────────────────────────┼────────────────────────────────────┘
                                                         │
                                                         ▼
                                                    render.py
                                                         │
                                                         ▼
                                               HTML Visualization
                                                         │
                                                         ▼
                                              tools/evaluate.py
                                                         │
                                                         ▼
                                               P/R/F1 Metrics
```

### CLI Mode (`cli.py`)
```
Command Line Arguments → cli.py → Load Environment (env.py)
                                        │
                                        ▼
                                  Check Ollama (ollama.py)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              baseline.py         pipeline.py              both
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        │
                                        ▼
                                   render.py
                                        │
                                        ▼
                               Output: JSON + HTML
```

---

## Dependencies

- **PyQt6** / **PyQt6-WebEngine** - GUI framework
- **pandas** - Data manipulation
- **requests** - HTTP client for Ollama API
- **matplotlib** - Evaluation charts
- **Ollama** - Local LLM server (external)
