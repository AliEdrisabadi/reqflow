from __future__ import annotations

import os
import sys
import json
import html as _html
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QCoreApplication
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QGridLayout,
    QSizePolicy,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except Exception:
    QWebEngineView = None
    HAS_WEBENGINE = False

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ----------------------------
# Paths / imports (project layout: ROOT/{src,prompts,data})
# ----------------------------
ROOT = Path(__file__).resolve().parent

SRC_DIR = (ROOT / "src").resolve()
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env import load_dotenv, find_dotenv
from ollama import ollama_check
import baseline as baseline_mod
import pipeline as pipeline_mod

# Load .env (search upward from ROOT)
load_dotenv(find_dotenv(str(ROOT)))


# ----------------------------
# Constants
# ----------------------------
APP_NAME = "ReqFlow"

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

TAG_COLORS_DEFAULT: Dict[str, str] = {
    "Main_actor": "#66c2ff",
    "Entity": "#a78bfa",
    "Action": "#2bd98f",
    "System_response": "#ffb020",
    "Condition": "#f472b6",
    "Precondition": "#60a5fa",
    "Constraint": "#fb7185",
    "Exception": "#f59e0b",
}


# ----------------------------
# .env
# ----------------------------
def _env_path(key: str, default_rel: str) -> Path:
    raw = (os.getenv(key) or "").strip()
    p = Path(raw) if raw else Path(default_rel)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


PROMPTS_DIR = _env_path("REQFLOW_PROMPTS_DIR", "prompts")
RESULTS_DIR = _env_path("REQFLOW_RESULTS_DIR", "result_gui")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASET = _env_path("REQFLOW_DATASET", "data/requirements_dataset.csv")
DEFAULT_GOLD = _env_path("REQFLOW_GOLD", "data/gold_annotations_seat_based.json")

HTML_THEME = (os.getenv("REQFLOW_HTML_THEME") or "dark").strip() or "dark"


def _variant_to_key(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ("0", "zero", "zero-shot", "zeroshot"):
        return "zero"
    if v in ("1", "one", "one-shot", "oneshot"):
        return "one"
    if v in ("few", "few-shot", "fewshot", "kshot"):
        return "few"
    return "zero"


def _prompt_env_key(kind: str, variant_key: str) -> str:
    return f"REQFLOW_{kind}_PROMPT_{variant_key.upper()}"


def _resolve_prompt_by_variant(kind: str, variant: str) -> Path:
    """
    kind: BASELINE | SEGMENT | TAG
    variant: zero/one/few (and aliases)
    """
    vk = _variant_to_key(variant)
    env_key = _prompt_env_key(kind, vk)
    name = (os.getenv(env_key) or "").strip()
    if not name:
        raise FileNotFoundError(
            f"Missing {env_key} in .env. Set it to a file under {PROMPTS_DIR} (or absolute path)."
        )

    p = Path(name)
    if not p.is_absolute():
        p = PROMPTS_DIR / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Prompt not found for {env_key}: {p}")
    return p


# ----------------------------
# GUI helpers
# ----------------------------
def safe_preview(text: str, n: int = 110) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t if len(t) <= n else (t[:n] + "...")


def set_table_from_df(table: QTableWidget, df: pd.DataFrame):
    table.setSortingEnabled(False)
    table.clear()
    table.setRowCount(0)
    table.setColumnCount(0)

    if df is None or df.empty:
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Info"])
        table.setRowCount(1)
        it = QTableWidgetItem("No data")
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(0, 0, it)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setSortingEnabled(False)
        return

    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    table.setRowCount(len(df))

    for r in range(len(df)):
        table.setRowHeight(r, 28)
        for c, col in enumerate(df.columns):
            val = df.iloc[r][col]
            item = QTableWidgetItem("" if pd.isna(val) else str(val))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if isinstance(val, (int, float)) and col not in ("tag",):
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)
            else:
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            table.setItem(r, c, item)

    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    header.setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setSortingEnabled(True)


# ----------------------------
# Render 
# ----------------------------
def _pct(x: float) -> str:
    return f"{x * 100.0:.2f}%"


def _bars_for_tag(text: str, spans: List[Dict[str, Any]], tag_colors: Dict[str, str]) -> str:
    n = max(1, len(text))
    bars: List[str] = []
    for sp in spans:
        st = int(sp.get("start", 0))
        en = int(sp.get("end", 0))
        tag = str(sp.get("tag", ""))
        seg = str(sp.get("text", ""))

        left = max(0.0, min(1.0, st / n))
        width = max(0.0, min(1.0, (en - st) / n))

        color = tag_colors.get(tag, "#999999")
        tip = _html.escape(f"{tag}: {seg}")
        bars.append(
            f'<div class="bar" style="left:{_pct(left)};width:{_pct(width)};background:{color};" title="{tip}"></div>'
        )
    return "\n".join(bars)


def _render_item(item: Dict[str, Any], tag_colors: Dict[str, str]) -> str:
    rid = item.get("id", "")
    text = str(item.get("text", ""))
    spans = item.get("spans", []) or []

    by_tag: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TAGS}
    for sp in spans:
        tag = sp.get("tag")
        if tag in by_tag:
            by_tag[tag].append(sp)

    rows = []
    for tag in TAGS:
        bars = _bars_for_tag(text, by_tag[tag], tag_colors=tag_colors)
        rows.append(
            f"""
            <div class="tagRow">
              <div class="tagLabel">{_html.escape(tag)}</div>
              <div class="track">{bars}</div>
            </div>
            """
        )

    return f"""
    <section class="card">
      <div class="cardHeader">
        <div class="rid">ID {_html.escape(str(rid))}</div>
        <div class="reqText">{_html.escape(text)}</div>
      </div>
      <div class="rows">
        {''.join(rows)}
      </div>
    </section>
    """


def render_spans_html(
    pred_json: Path,
    out_html: Path,
    *,
    theme: str = "dark",
    tag_colors: Optional[Dict[str, str]] = None,
) -> None:
    items = json.loads(pred_json.read_text(encoding="utf-8"))
    tag_colors = tag_colors or dict(TAG_COLORS_DEFAULT)

    is_dark = (theme or "dark").lower().strip() != "light"

    bg = "#15181c" if is_dark else "#ffffff"
    card = "#0f1216" if is_dark else "#f7f7f7"
    textc = "#eaeaea" if is_dark else "#222222"
    mutec = "#a8b0bb" if is_dark else "#555555"
    border = "#2a2f36" if is_dark else "#e0e0e0"
    track = "#1b2026" if is_dark else "#ffffff"

    css = f"""
    :root {{
      --bg: {bg};
      --card: {card};
      --text: {textc};
      --muted: {mutec};
      --border: {border};
      --track: {track};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background: var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 1280px;
      margin: 16px auto;
      padding: 0 12px 28px 12px;
    }}
    .header {{
      display:flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin: 8px 0 18px 0;
    }}
    .header h1 {{
      font-size: 18px;
      margin: 0;
      font-weight: 800;
      letter-spacing: 0.2px;
    }}
    .header .meta {{
      font-size: 12px;
      color: var(--muted);
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      margin: 12px 0;
      box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }}
    .cardHeader {{
      margin-bottom: 10px;
    }}
    .rid {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .reqText {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .rows {{
      margin-top: 12px;
      display: grid;
      gap: 8px;
    }}
    .tagRow {{
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 10px;
      align-items: center;
    }}
    .tagLabel {{
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .track {{
      position: relative;
      height: 18px;
      border-radius: 10px;
      background: var(--track);
      border: 1px solid var(--border);
      overflow: hidden;
    }}
    .bar {{
      position: absolute;
      top: 1px;
      bottom: 1px;
      border-radius: 9px;
      opacity: 0.92;
    }}
    .bar:hover {{
      opacity: 1.0;
      filter: saturate(1.1);
    }}
    """

    body = "\n".join(_render_item(it, tag_colors=tag_colors) for it in items)

    html_doc = f"""<!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>ReqFlow — Spans</title>
        <style>{css}</style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>ReqFlow — Span Visualization</h1>
            <div class="meta">{_html.escape(pred_json.name)}</div>
          </div>
          {body}
        </div>
      </body>
    </html>
    """
    out_html.write_text(html_doc, encoding="utf-8")


# ----------------------------
# Evaluation logic 
# ----------------------------
@dataclass(frozen=True)
class Span:
    tag: str
    start: int
    end: int
    text: str


def _to_spans(item: dict) -> List[Span]:
    spans: List[Span] = []
    for sp in (item.get("spans", []) or []):
        tag = sp.get("tag")
        st_i = sp.get("start")
        en_i = sp.get("end")
        tx = sp.get("text", "")
        if tag in TAGS and isinstance(st_i, int) and isinstance(en_i, int) and en_i >= st_i:
            spans.append(Span(tag=tag, start=st_i, end=en_i, text=str(tx)))
    return spans


def load_gold(path: Path) -> Dict[int, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {int(it.get("id")): it for it in raw if isinstance(it, dict) and "id" in it}
    if isinstance(raw, dict):
        if "items" in raw and isinstance(raw["items"], list):
            return {int(it.get("id")): it for it in raw["items"] if isinstance(it, dict) and "id" in it}
        if "data" in raw and isinstance(raw["data"], list):
            return {int(it.get("id")): it for it in raw["data"] if isinstance(it, dict) and "id" in it}
        out: Dict[int, dict] = {}
        for k, v in raw.items():
            try:
                rid = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                out[rid] = v
        if out:
            return out
    raise ValueError("Unsupported gold JSON format.")


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
    """
    Compute match score for two spans.
    
    Mode "Exact": requires exact boundary match
    Mode "Relaxed": uses max(IoU, TextSimilarity) to handle LLM boundary variance
    """
    if a.tag != b.tag:
        return 0.0
    if mode == "Exact":
        return 1.0 if (a.start == b.start and a.end == b.end) else 0.0
    # Relaxed: max(IoU, TextSimilarity) - matches evaluation.py logic
    s_iou = iou(a, b)
    s_txt = text_sim(a, b)
    return max(s_iou, s_txt)


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


def evaluate_run(pred_items: List[dict], gold_map: Dict[int, dict], mode: str, threshold: float) -> Tuple[dict, pd.DataFrame]:
    """
    Evaluate predictions against gold annotations.
    
    Returns:
        summary: dict with Micro F1 (primary metric) and Macro F1 (secondary)
        df: DataFrame with per-tag metrics
    """
    pred_map = {int(it.get("id")): it for it in pred_items if "id" in it}
    common_ids = sorted(set(pred_map.keys()).intersection(set(gold_map.keys())))

    overall_tp = overall_fp = overall_fn = 0
    rows: List[Dict[str, Any]] = []
    tag_f1_scores: List[float] = []  # For Macro F1 computation

    for tag in TAGS:
        tp_t = fp_t = fn_t = 0
        for rid in common_ids:
            pred_item = pred_map.get(rid)
            gold_item = gold_map.get(rid)
            if pred_item is None or gold_item is None:
                continue

            pred_spans = [s for s in _to_spans(pred_item) if s.tag == tag]
            gold_spans = [s for s in _to_spans(gold_item) if s.tag == tag]

            tp, fp, fn = match_counts(pred_spans, gold_spans, mode=mode, threshold=threshold)
            tp_t += tp
            fp_t += fp
            fn_t += fn

        p, r, f1 = prf(tp_t, fp_t, fn_t)
        tag_f1_scores.append(f1)  # Collect F1 for Macro computation
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

    # Micro F1: computed from aggregated TP/FP/FN (PRIMARY METRIC)
    p_all, r_all, f1_micro = prf(overall_tp, overall_fp, overall_fn)
    
    # Macro F1: average of per-tag F1 scores (SECONDARY METRIC)
    f1_macro = sum(tag_f1_scores) / len(tag_f1_scores) if tag_f1_scores else 0.0
    
    summary = {
        "common_requirements": len(common_ids),
        "mode": mode,
        "threshold": threshold,
        "TP": overall_tp,
        "FP": overall_fp,
        "FN": overall_fn,
        "precision": round(p_all, 4),
        "recall": round(r_all, 4),
        "f1": round(f1_micro, 4),           # Micro F1 (primary)
        "micro_f1": round(f1_micro, 4),     # Explicit Micro F1
        "macro_f1": round(f1_macro, 4),     # Macro F1 (secondary)
    }

    df = pd.DataFrame(rows).sort_values(by="f1", ascending=False)
    return summary, df


def threshold_curve(pred_items: List[dict], gold_map: Dict[int, dict], mode: str) -> pd.DataFrame:
    if mode == "Exact":
        s, _ = evaluate_run(pred_items, gold_map, mode=mode, threshold=1.0)
        return pd.DataFrame(
            [
                {"threshold": 1.0, "metric": "precision", "value": s["precision"]},
                {"threshold": 1.0, "metric": "recall", "value": s["recall"]},
                {"threshold": 1.0, "metric": "f1", "value": s["f1"]},
            ]
        )

    ths = [round(i / 20, 2) for i in range(0, 21)]
    out_rows: List[Dict[str, Any]] = []
    for t in ths:
        s, _ = evaluate_run(pred_items, gold_map, mode=mode, threshold=t)
        out_rows.append({"threshold": t, "metric": "precision", "value": s["precision"]})
        out_rows.append({"threshold": t, "metric": "recall", "value": s["recall"]})
        out_rows.append({"threshold": t, "metric": "f1", "value": s["f1"]})
    return pd.DataFrame(out_rows)


def _apply_dark_axes(ax):
    ax.set_facecolor("#0f1216")
    ax.figure.set_facecolor("#0f1216")
    ax.tick_params(colors="#dbe2ea")
    ax.xaxis.label.set_color("#dbe2ea")
    ax.yaxis.label.set_color("#dbe2ea")
    ax.title.set_color("#f2f6fb")
    for spine in ax.spines.values():
        spine.set_color("#2a2f36")
    ax.grid(True, color="#242a31", alpha=0.9)


class MetricLineChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(6, 3), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def plot_threshold_curve(self, df: pd.DataFrame, title: str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#dbe2ea")
            _apply_dark_axes(ax)
            self.canvas.draw()
            return

        for metric in ["precision", "recall", "f1"]:
            d2 = df[df["metric"] == metric]
            ax.plot(d2["threshold"], d2["value"], marker="o", label=metric)

        ax.set_title(title, pad=12)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        _apply_dark_axes(ax)

        self.fig.tight_layout()
        self.canvas.draw()


class ComparisonChart(QWidget):
    """Chart comparing Relaxed Match vs Exact Match across thresholds"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(6, 4), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def plot_comparison(self, pred_items: List[dict], gold_map: Dict[int, dict]):
        """Compare Relaxed Match vs Exact Match for Micro and Macro F1"""
        self.fig.clear()
        
        if not pred_items or not gold_map:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color="#dbe2ea")
            _apply_dark_axes(ax)
            self.canvas.draw()
            return

        # Create 2 subplots: Micro F1 and Macro F1
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)

        # Compute metrics across thresholds
        thresholds = [round(i / 20, 2) for i in range(0, 21)]
        
        relaxed_micro = []
        relaxed_macro = []
        
        for t in thresholds:
            summ, _ = evaluate_run(pred_items, gold_map, mode="Relaxed", threshold=t)
            relaxed_micro.append(summ["micro_f1"])
            relaxed_macro.append(summ["macro_f1"])
        
        # Exact match (single point at threshold 1.0)
        exact_summ, _ = evaluate_run(pred_items, gold_map, mode="Exact", threshold=1.0)
        exact_micro = exact_summ["micro_f1"]
        exact_macro = exact_summ["macro_f1"]

        # Plot Micro F1 comparison
        ax1.plot(thresholds, relaxed_micro, marker="o", linewidth=2, 
                label="Relaxed Match", color="#2bd98f")
        ax1.axhline(y=exact_micro, color="#fb7185", linestyle="--", linewidth=2,
                   label=f"Exact Match (F1={exact_micro:.3f})")
        ax1.axvline(x=0.5, color="#ffb020", linestyle=":", alpha=0.5, linewidth=1.5)
        ax1.text(0.5, ax1.get_ylim()[1] * 0.95, "Recommended\nthreshold=0.5", 
                ha="center", va="top", color="#ffb020", fontsize=8)
        ax1.set_title("Micro F1 Comparison (PRIMARY METRIC)", pad=8, fontsize=10, fontweight="bold")
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Micro F1")
        ax1.set_ylim(0.0, 1.0)
        ax1.legend(loc="lower left", fontsize=8)
        ax1.grid(True, alpha=0.3)
        _apply_dark_axes(ax1)

        # Plot Macro F1 comparison
        ax2.plot(thresholds, relaxed_macro, marker="o", linewidth=2,
                label="Relaxed Match", color="#66c2ff")
        ax2.axhline(y=exact_macro, color="#f472b6", linestyle="--", linewidth=2,
                   label=f"Exact Match (F1={exact_macro:.3f})")
        ax2.axvline(x=0.5, color="#ffb020", linestyle=":", alpha=0.5, linewidth=1.5)
        ax2.set_title("Macro F1 Comparison (SECONDARY METRIC)", pad=8, fontsize=10, fontweight="bold")
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Macro F1")
        ax2.set_ylim(0.0, 1.0)
        ax2.legend(loc="lower left", fontsize=8)
        ax2.grid(True, alpha=0.3)
        _apply_dark_axes(ax2)

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


# ----------------------------
# Worker thread
# ----------------------------
class WorkerThread(QThread):
    # html_path, run_dir, pred_json_path, pred_label
    finished = pyqtSignal(str, str, str, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        mode: str,
        ids_csv: str,
        dataset_path: Path,
        out_dir: Path,
        baseline_variant: str,
        pipeline_variant: str,
    ):
        super().__init__()
        self.mode = mode
        self.ids_csv = ids_csv
        self.dataset_path = dataset_path
        self.out_dir = out_dir
        self.baseline_variant = _variant_to_key(baseline_variant)
        self.pipeline_variant = _variant_to_key(pipeline_variant)

    def run(self):
        try:
            html_path = ""
            pred_json_path = ""
            pred_label = ""

            if self.mode in ("Baseline", "Both"):
                b_prompt = _resolve_prompt_by_variant("BASELINE", self.baseline_variant)

                bdir = self.out_dir / "baseline" / self.baseline_variant
                bdir.mkdir(parents=True, exist_ok=True)
                bj = bdir / "baseline_spans.json"
                bh = bdir / "baseline_spans.html"

                baseline_mod.main(
                    str(self.dataset_path),
                    str(bj),
                    None,
                    str(b_prompt),
                    self.ids_csv,
                )
                render_spans_html(bj, bh, theme=HTML_THEME, tag_colors=TAG_COLORS_DEFAULT)

                html_path = str(bh.resolve())
                pred_json_path = str(bj.resolve())
                pred_label = f"Baseline ({self.baseline_variant})"

            if self.mode in ("Pipeline", "Both"):
                s_prompt = _resolve_prompt_by_variant("SEGMENT", self.pipeline_variant)
                t_prompt = _resolve_prompt_by_variant("TAG", self.pipeline_variant)

                pdir = self.out_dir / "pipeline" / self.pipeline_variant
                pdir.mkdir(parents=True, exist_ok=True)
                pj = pdir / "pipeline_spans.json"
                ph = pdir / "pipeline_spans.html"

                pipeline_mod.main(
                    str(self.dataset_path),
                    str(pj),
                    None,
                    self.ids_csv,
                    segment_prompt_path=str(s_prompt),
                    tag_prompt_path=str(t_prompt),
                )
                render_spans_html(pj, ph, theme=HTML_THEME, tag_colors=TAG_COLORS_DEFAULT)

                # Prefer pipeline for display/eval if it ran
                html_path = str(ph.resolve())
                pred_json_path = str(pj.resolve())
                pred_label = f"Pipeline ({self.pipeline_variant})"

            self.finished.emit(html_path, str(self.out_dir.resolve()), pred_json_path, pred_label)

        except Exception as e:
            self.error.emit(str(e))


# ----------------------------
# Main GUI
# ----------------------------
class ReqFlowApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1460, 920)

        self.dataset_path: Path = DEFAULT_DATASET if DEFAULT_DATASET.exists() else Path()
        self.df_dataset: Optional[pd.DataFrame] = None

        self.gold_path: Path = DEFAULT_GOLD if DEFAULT_GOLD.exists() else Path()
        self.current_run_dir: Optional[Path] = None
        self.current_pred_json: Optional[Path] = None
        self.current_pred_label: str = ""
        self.worker: Optional[WorkerThread] = None

        self.tag_colors: Dict[str, str] = dict(TAG_COLORS_DEFAULT)

        self._build_ui()
        self._apply_styles()

        if self.dataset_path.exists():
            self.dataset_label.setText(str(self.dataset_path))
            self.load_dataset()
        else:
            self.dataset_label.setText("Select dataset CSV (id, text_en).")

        if self.gold_path.exists():
            self.gold_label.setText(str(self.gold_path))
        else:
            self.gold_label.setText("Select gold JSON (optional for evaluation).")

        self._set_eval_empty_state()
        self._refresh_legend_ui()
        self._on_method_changed()

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background: #15181c;
                color: #eaeaea;
                font-family: "Segoe UI";
                font-size: 12px;
            }
            QMainWindow { background: #15181c; }

            QTabWidget::pane {
                border: 1px solid #222831;
                border-radius: 10px;
                padding: 6px;
                background: #15181c;
            }
            QTabBar::tab {
                background: #0f1216;
                border: 1px solid #2a2f36;
                border-bottom: none;
                padding: 10px 14px;
                margin-right: 6px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                color: #cfd6dd;
                font-weight: 700;
            }
            QTabBar::tab:selected {
                background: #1b2026;
                color: #ffffff;
                border-color: #3a4656;
            }

            QGroupBox {
                background: #14181d;
                border: 1px solid #2a2f36;
                border-radius: 12px;
                margin-top: 12px;
                padding: 12px;
                color: #dbe2ea;
                font-weight: 800;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                color: #f2f6fb;
            }

            QListWidget, QTableWidget {
                background: #0f1216;
                border: 1px solid #2a2f36;
                border-radius: 12px;
                color: #eaeaea;
                alternate-background-color: #12161b;
                selection-background-color: #2b90d9;
                selection-color: #ffffff;
                gridline-color: #1f252c;
            }

            QHeaderView::section {
                background: #1b2026;
                color: #f2f6fb;
                border: 0px;
                padding: 8px;
                font-weight: 900;
            }

            QComboBox {
                background: #0f1216;
                border: 1px solid #2a2f36;
                border-radius: 12px;
                padding: 7px 10px;
                color: #eaeaea;
                font-weight: 700;
            }

            QPushButton {
                background: #2b90d9;
                border: 0px;
                border-radius: 12px;
                padding: 11px 14px;
                color: white;
                font-weight: 900;
            }
            QPushButton:hover { background: #257fbe; }
            QPushButton:pressed { background: #1f6ea3; }
            QPushButton:disabled { background: #3a414a; color: #b3bbc4; }

            QPushButton#analysisButton { background: #27c08a; }
            QPushButton#analysisButton:hover { background: #22ad7b; }
            QPushButton#analysisButton:pressed { background: #1e9a6d; }
            QPushButton#analysisButton:disabled { background: #3a414a; color: #b3bbc4; }

            QProgressBar {
                background: #0f1216;
                border: 1px solid #2a2f36;
                border-radius: 12px;
                text-align: center;
                color: #eaeaea;
                height: 14px;
                font-weight: 800;
            }
            QProgressBar::chunk {
                background: #2b90d9;
                border-radius: 12px;
            }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        # ---- Left panel
        left = QFrame()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(12)

        title = QLabel(APP_NAME)
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        left_layout.addWidget(title)

        ds_group = QGroupBox("Dataset")
        ds_layout = QVBoxLayout(ds_group)
        self.dataset_label = QLabel("")
        self.dataset_label.setWordWrap(True)
        ds_layout.addWidget(self.dataset_label)

        btn_browse = QPushButton("Browse CSV…")
        btn_browse.clicked.connect(self.browse_dataset)
        ds_layout.addWidget(btn_browse)
        left_layout.addWidget(ds_group)

        left_layout.addWidget(QLabel("Select requirements (checkbox):"))
        self.req_list = QListWidget()
        self.req_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.req_list.itemChanged.connect(self._update_checked_count)
        left_layout.addWidget(self.req_list, stretch=1)

        footer_row = QHBoxLayout()
        self.checked_label = QLabel("Checked: 0")
        self.checked_label.setStyleSheet("color: #b9c3cf; font-weight: 700;")
        footer_row.addWidget(self.checked_label)
        footer_row.addStretch(1)

        self.btn_check_all = QPushButton("Check all")
        self.btn_check_all.clicked.connect(self.check_all)
        footer_row.addWidget(self.btn_check_all)

        self.btn_uncheck_all = QPushButton("Uncheck all")
        self.btn_uncheck_all.clicked.connect(self.uncheck_all)
        footer_row.addWidget(self.btn_uncheck_all)

        left_layout.addLayout(footer_row)

        run_group = QGroupBox("Run")
        run_layout = QVBoxLayout(run_group)

        run_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Baseline", "Pipeline", "Both"])
        self.method_combo.setCurrentText("Both")
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        run_layout.addWidget(self.method_combo)

        # ----------------------------
        # Prompt variants UI 
        # ----------------------------
        def _make_variant_combo(default_key: str) -> QComboBox:
            cb = QComboBox()
            cb.addItem("Zero-shot", "zero")
            cb.addItem("One-shot", "one")
            cb.addItem("Few-shot", "few")
            dk = _variant_to_key(default_key)
            idx = cb.findData(dk)
            cb.setCurrentIndex(idx if idx >= 0 else 0)
            return cb

        # --- Single variant row 
        self.single_variant_row = QWidget()
        sv_lay = QHBoxLayout(self.single_variant_row)
        sv_lay.setContentsMargins(0, 0, 0, 0)
        sv_lay.setSpacing(8)

        sv_lay.addWidget(QLabel("Prompt variant:"))
        self.prompt_variant_combo = _make_variant_combo(os.getenv("REQFLOW_DEFAULT_BASELINE_VARIANT", "zero"))
        sv_lay.addWidget(self.prompt_variant_combo)

        run_layout.addWidget(self.single_variant_row)

        # --- Baseline row (ONLY for Both)
        self.baseline_variant_row = QWidget()
        bv_lay = QHBoxLayout(self.baseline_variant_row)
        bv_lay.setContentsMargins(0, 0, 0, 0)
        bv_lay.setSpacing(8)

        bv_lay.addWidget(QLabel("Baseline variant:"))
        self.baseline_prompt_combo = _make_variant_combo(os.getenv("REQFLOW_DEFAULT_BASELINE_VARIANT", "zero"))
        bv_lay.addWidget(self.baseline_prompt_combo)

        run_layout.addWidget(self.baseline_variant_row)

        # --- Pipeline row (ONLY for Both)
        self.pipeline_variant_row = QWidget()
        pv_lay = QHBoxLayout(self.pipeline_variant_row)
        pv_lay.setContentsMargins(0, 0, 0, 0)
        pv_lay.setSpacing(8)

        pv_lay.addWidget(QLabel("Pipeline variant:"))
        self.pipeline_prompt_combo = _make_variant_combo(os.getenv("REQFLOW_DEFAULT_PIPELINE_VARIANT", "zero"))
        pv_lay.addWidget(self.pipeline_prompt_combo)

        run_layout.addWidget(self.pipeline_variant_row)

        self.run_btn = QPushButton("Analysis")
        self.run_btn.setObjectName("analysisButton")
        self.run_btn.setFixedHeight(44)
        self.run_btn.clicked.connect(self.run_analysis)
        run_layout.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        run_layout.addWidget(self.progress)

        left_layout.addWidget(run_group)
        splitter.addWidget(left)

        # ---- Right panel
        right = QFrame()
        right_layout = QVBoxLayout(right)

        self.tabs = QTabWidget()

        # Main tab
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.legend_frame = QFrame()
        self.legend_frame.setStyleSheet("""
            QFrame {
                background: #0f1216;
                border: 1px solid #2a2f36;
                border-radius: 12px;
            }
        """)
        self.legend_layout = QGridLayout(self.legend_frame)
        self.legend_layout.setContentsMargins(12, 10, 12, 10)
        self.legend_layout.setHorizontalSpacing(14)
        self.legend_layout.setVerticalSpacing(8)

        legend_title = QLabel("Legend (Tag colors):")
        legend_title.setStyleSheet("color: #f2f6fb; font-weight: 900;")
        self.legend_layout.addWidget(legend_title, 0, 0, 1, 4)

        self.legend_items: Dict[str, Tuple[QFrame, QLabel]] = {}
        row0 = 1
        idx = 0
        for t in TAGS:
            r = row0 + (idx // 4)
            c = idx % 4
            w = self._make_legend_item(t, TAG_COLORS_DEFAULT.get(t, "#888888"))
            self.legend_layout.addWidget(w, r, c)
            idx += 1

        main_layout.addWidget(self.legend_frame)

        if HAS_WEBENGINE:
            self.browser = QWebEngineView()
            main_layout.addWidget(self.browser, stretch=1)
            self._configure_webengine(self.browser)
        else:
            self.browser = None
            lab = QLabel("PyQt6-WebEngine is not installed.\nInstall: pip install PyQt6-WebEngine")
            lab.setWordWrap(True)
            main_layout.addWidget(lab, stretch=1)

        self.tabs.addTab(main_tab, "Main")

        # Evaluation tab
        eval_tab = QWidget()
        e_layout = QVBoxLayout(eval_tab)
        e_layout.setSpacing(10)

        gold_group = QGroupBox("Gold & Evaluation Settings")
        gold_layout = QVBoxLayout(gold_group)

        self.gold_label = QLabel("")
        self.gold_label.setWordWrap(True)
        gold_layout.addWidget(self.gold_label)

        row = QHBoxLayout()
        btn_gold = QPushButton("Browse gold JSON…")
        btn_gold.clicked.connect(self.browse_gold)
        row.addWidget(btn_gold)

        row.addSpacing(10)
        row.addWidget(QLabel("Mode:"))
        self.eval_mode = QComboBox()
        self.eval_mode.addItems(["Exact", "Relaxed"])
        self.eval_mode.setCurrentText("Relaxed")
        self.eval_mode.currentIndexChanged.connect(self._on_eval_mode_changed)
        row.addWidget(self.eval_mode)

        row.addSpacing(10)
        row.addWidget(QLabel("Threshold:"))
        self.eval_thr = QComboBox()
        self.eval_thr.addItems([f"{i/20:.2f}" for i in range(0, 21)])
        self.eval_thr.setCurrentText("0.50")  # Recommended threshold for relaxed matching
        row.addWidget(self.eval_thr)

        row.addStretch(1)

        self.eval_btn = QPushButton("Compute P/R/F1")
        self.eval_btn.clicked.connect(self.compute_evaluation)
        row.addWidget(self.eval_btn)

        gold_layout.addLayout(row)
        e_layout.addWidget(gold_group)

        self.eval_summary = QLabel("No evaluation yet.")
        self.eval_summary.setWordWrap(True)
        self.eval_summary.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        e_layout.addWidget(self.eval_summary)

        # Create tab widget for different evaluation views
        eval_tabs = QTabWidget()
        
        # Tab 1: Per-tag metrics table + threshold curve
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        
        eval_split = QSplitter(Qt.Orientation.Horizontal)
        self.eval_table = QTableWidget()
        eval_split.addWidget(self.eval_table)

        self.eval_curve = MetricLineChart()
        eval_split.addWidget(self.eval_curve)

        eval_split.setStretchFactor(0, 1)
        eval_split.setStretchFactor(1, 1)
        
        metrics_layout.addWidget(eval_split)
        eval_tabs.addTab(metrics_tab, "Per-Tag Metrics")
        
        # Tab 2: Comparison chart (Relaxed vs Exact)
        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        
        comparison_info = QLabel(
            "This chart compares Relaxed Match (with IoU/TextSim threshold) vs Exact Match "
            "for both Micro F1 (primary metric) and Macro F1 (secondary metric).\n"
            "Recommended: Use Relaxed Match with threshold ≥ 0.5 to handle LLM boundary variance."
        )
        comparison_info.setWordWrap(True)
        comparison_info.setStyleSheet("color: #b9c3cf; padding: 8px;")
        comparison_layout.addWidget(comparison_info)
        
        self.comparison_chart = ComparisonChart()
        comparison_layout.addWidget(self.comparison_chart, stretch=1)
        
        eval_tabs.addTab(comparison_tab, "Relaxed vs Exact Comparison")

        e_layout.addWidget(eval_tabs, stretch=1)
        self.tabs.addTab(eval_tab, "Evaluation")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1040])

    def _on_method_changed(self, *_):
        mode = self.method_combo.currentText()
        is_both = (mode == "Both")

        # Show a single variant selector for Baseline/Pipeline; show two selectors only for Both
        self.single_variant_row.setVisible(not is_both)
        self.baseline_variant_row.setVisible(is_both)
        self.pipeline_variant_row.setVisible(is_both)

        # Sync: when leaving Both, keep the single dropdown aligned with the method's last selection
        if not is_both:
            if mode == "Baseline":
                v = self.baseline_prompt_combo.currentData() or "zero"
            else:  # Pipeline
                v = self.pipeline_prompt_combo.currentData() or "zero"

            v = _variant_to_key(str(v))
            idx = self.prompt_variant_combo.findData(v)
            if idx >= 0:
                self.prompt_variant_combo.setCurrentIndex(idx)

    def _make_legend_item(self, tag: str, color_hex: str) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        sw = QFrame()
        sw.setFixedSize(14, 14)
        sw.setStyleSheet(
            f"background: {color_hex}; border-radius: 7px; border: 1px solid rgba(255,255,255,0.18);"
        )
        sw.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        lbl = QLabel(tag)
        lbl.setStyleSheet("color: #eaeaea; font-weight: 800;")
        lbl.setToolTip(f"{tag} tag")

        lay.addWidget(sw)
        lay.addWidget(lbl)
        lay.addStretch(1)

        self.legend_items[tag] = (sw, lbl)
        return w

    def _refresh_legend_ui(self):
        for t in TAGS:
            sw, _lbl = self.legend_items.get(t, (None, None))
            if sw is not None:
                col = TAG_COLORS_DEFAULT.get(t, "#888888")
                sw.setStyleSheet(
                    f"background: {col}; border-radius: 7px; border: 1px solid rgba(255,255,255,0.18);"
                )

    def _configure_webengine(self, view):
        try:
            from PyQt6.QtWebEngineCore import QWebEngineSettings
            s = view.settings()
            for name in ["LocalContentCanAccessFileUrls", "LocalContentCanAccessRemoteUrls"]:
                if hasattr(QWebEngineSettings.WebAttribute, name):
                    s.setAttribute(getattr(QWebEngineSettings.WebAttribute, name), True)
        except Exception:
            pass

    def _set_eval_empty_state(self):
        set_table_from_df(self.eval_table, pd.DataFrame())
        self.eval_curve.plot_threshold_curve(pd.DataFrame(), "Threshold sweep (Precision/Recall/F1)")
        # Clear comparison chart on initialization
        if hasattr(self, 'comparison_chart'):
            self.comparison_chart.plot_comparison([], {})

    def _checked_ids(self) -> List[int]:
        ids: List[int] = []
        for i in range(self.req_list.count()):
            it = self.req_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                rid = it.data(Qt.ItemDataRole.UserRole)
                try:
                    ids.append(int(rid))
                except Exception:
                    pass
        return ids

    def _update_checked_count(self, _item=None):
        self.checked_label.setText(f"Checked: {len(self._checked_ids())}")

    def check_all(self):
        if self.req_list.count() == 0:
            return
        self.req_list.blockSignals(True)
        try:
            for i in range(self.req_list.count()):
                it = self.req_list.item(i)
                it.setCheckState(Qt.CheckState.Checked)
        finally:
            self.req_list.blockSignals(False)
            self._update_checked_count()

    def uncheck_all(self):
        if self.req_list.count() == 0:
            return
        self.req_list.blockSignals(True)
        try:
            for i in range(self.req_list.count()):
                it = self.req_list.item(i)
                it.setCheckState(Qt.CheckState.Unchecked)
        finally:
            self.req_list.blockSignals(False)
            self._update_checked_count()

    def browse_dataset(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Dataset CSV", str(ROOT), "CSV Files (*.csv)")
        if fname:
            self.dataset_path = Path(fname).resolve()
            self.dataset_label.setText(str(self.dataset_path))
            self.load_dataset()

    def load_dataset(self):
        self.req_list.blockSignals(True)
        try:
            self.req_list.clear()
            if not self.dataset_path.exists():
                QMessageBox.warning(self, "Dataset missing", "Please select a valid dataset CSV.")
                return

            try:
                df = pd.read_csv(self.dataset_path)
            except Exception as e:
                QMessageBox.critical(self, "CSV read error", str(e))
                return

            if "id" not in df.columns or "text_en" not in df.columns:
                QMessageBox.warning(self, "Invalid CSV", "CSV must have columns: id, text_en")
                return

            self.df_dataset = df.copy()
            self.df_dataset["id"] = self.df_dataset["id"].astype(int)

            for _, row in self.df_dataset.iterrows():
                rid = int(row["id"])
                text = str(row["text_en"])
                it = QListWidgetItem(f"{rid}: {safe_preview(text, 120)}")
                it.setData(Qt.ItemDataRole.UserRole, rid)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Unchecked)
                self.req_list.addItem(it)

        finally:
            self.req_list.blockSignals(False)
            self._update_checked_count()

    def run_analysis(self):
        ok, msg = ollama_check(timeout=5)
        if not ok:
            QMessageBox.critical(self, "Ollama error", msg)
            return

        if not self.dataset_path.exists():
            QMessageBox.warning(self, "Dataset missing", "Please select a dataset first.")
            return

        checked_ids = self._checked_ids()
        if not checked_ids:
            QMessageBox.warning(self, "Select IDs", "Tick at least one requirement (checkbox).")
            return

        mode = self.method_combo.currentText()

        # If not Both: one dropdown controls the method's prompt variant.
        # If Both: separate dropdowns exist.
        if mode == "Both":
            bvar = self.baseline_prompt_combo.currentData() or "zero"
            pvar = self.pipeline_prompt_combo.currentData() or "zero"
        else:
            v = self.prompt_variant_combo.currentData() or "zero"
            bvar = v
            pvar = v

        # Validate prompts exist (fail fast with a clear message)
        try:
            if mode in ("Baseline", "Both"):
                _ = _resolve_prompt_by_variant("BASELINE", str(bvar))
            if mode in ("Pipeline", "Both"):
                _ = _resolve_prompt_by_variant("SEGMENT", str(pvar))
                _ = _resolve_prompt_by_variant("TAG", str(pvar))
        except Exception as e:
            QMessageBox.critical(self, "Prompt configuration error", str(e))
            return

        ids_csv = ",".join(str(x) for x in checked_ids)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (RESULTS_DIR / f"run_{run_id}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        self.run_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)

        self.worker = WorkerThread(
            mode,
            ids_csv,
            self.dataset_path,
            out_dir,
            baseline_variant=str(bvar),
            pipeline_variant=str(pvar),
        )
        self.worker.finished.connect(self.on_run_finished)
        self.worker.error.connect(self.on_run_error)
        self.worker.start()

    def on_run_finished(self, html_path: str, run_dir: str, pred_json_path: str, pred_label: str):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)

        self.current_run_dir = Path(run_dir).resolve()
        self.current_pred_json = Path(pred_json_path).resolve() if pred_json_path else None
        self.current_pred_label = pred_label or ""

        if self.browser is not None and html_path:
            html_file = Path(html_path).resolve()
            self.browser.setUrl(QUrl.fromLocalFile(str(html_file)))

        self.tabs.setCurrentIndex(0)

    def on_run_error(self, err: str):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Error", err)

    def browse_gold(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Gold JSON", str(ROOT), "JSON Files (*.json)")
        if fname:
            self.gold_path = Path(fname).resolve()
            self.gold_label.setText(str(self.gold_path))

    def _on_eval_mode_changed(self, _idx: int):
        mode = self.eval_mode.currentText()
        self.eval_thr.setEnabled(mode != "Exact")

    def compute_evaluation(self):
        if not self.current_pred_json or not self.current_pred_json.exists():
            QMessageBox.warning(self, "No predictions", "Run Analysis first (so prediction JSON exists).")
            return

        if not self.gold_path.exists():
            QMessageBox.warning(self, "No gold", "Select a gold JSON first.")
            return

        try:
            pred_items = json.loads(self.current_pred_json.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Predictions read error", str(e))
            return

        try:
            gold_map = load_gold(self.gold_path)
        except Exception as e:
            QMessageBox.critical(self, "Gold parse error", str(e))
            return

        mode = self.eval_mode.currentText()
        thr = 1.0 if mode == "Exact" else float(self.eval_thr.currentText())

        summ, df = evaluate_run(pred_items, gold_map, mode=mode, threshold=thr)
        set_table_from_df(self.eval_table, df)

        curve_df = threshold_curve(pred_items, gold_map, mode=mode)
        self.eval_curve.plot_threshold_curve(curve_df, "Threshold sweep (Precision/Recall/F1)")

        # Update comparison chart (Relaxed vs Exact for Micro/Macro F1)
        self.comparison_chart.plot_comparison(pred_items, gold_map)

        # Display evaluation summary with Micro F1 as PRIMARY metric
        metric_label = "PRIMARY" if mode == "Relaxed" and thr >= 0.5 else "COMPUTED"
        self.eval_summary.setText(
            f"Run: {self.current_run_dir.name if self.current_run_dir else ''} | Pred: {self.current_pred_label} | "
            f"Mode: {summ['mode']} | Threshold: {summ['threshold']}\n"
            f"Samples: {summ['common_requirements']} | TP: {summ['TP']} | FP: {summ['FP']} | FN: {summ['FN']}\n"
            f"Precision: {summ['precision']} | Recall: {summ['recall']} | "
            f"Micro F1: {summ['micro_f1']} ({metric_label}) | Macro F1: {summ['macro_f1']} (SECONDARY)"
        )


def _setup_webengine_stability_flags():
    if sys.platform.startswith("win"):
        os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-software-rasterizer")


if __name__ == "__main__":
    os.chdir(str(ROOT))
    _setup_webengine_stability_flags()

    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    app = QApplication(sys.argv)
    w = ReqFlowApp()
    w.show()
    sys.exit(app.exec())
