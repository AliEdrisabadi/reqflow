from __future__ import annotations

"""
ReqFlow GUI (LLM4SE A3)
- Baseline: single-agent extraction (NO segmentation)
- Pipeline: agent-chain with a Segmenter agent (preprocess) + specialist agents
- Evaluation: compares ONLY (tag, span_text) vs gold (offsets are ignored)
"""

import json
import os
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd

# -----------------------------------------------------------------------------
# Windows + QtWebEngine stability knobs (must be set BEFORE importing WebEngine)
# -----------------------------------------------------------------------------
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")

if os.name == "nt":
    # Avoid DirectComposition / GPU compositor issues on some Windows setups
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-gpu-compositing")
    os.environ.setdefault("QT_OPENGL", "software")

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QCoreApplication
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView

from src.env import load_dotenv
from src.baseline import run_baseline
from src.pipeline import run_pipeline
from src.render import main as render_html


# -----------------------------
# Slide taxonomy
# -----------------------------
TAGS = [
    "Purpose",
    "Trigger",
    "Precondition",
    "Condition",
    "Action",
    "System_response",
    "Entity",
    "Main_actor",
]

TAG_ALIASES = {
    "Constraint": "Condition",
    "Exception": "Trigger",
}


def normalize_tag(tag: Any) -> Optional[str]:
    if not isinstance(tag, str):
        return None
    tag = tag.strip()
    tag = TAG_ALIASES.get(tag, tag)
    return tag if tag in TAGS else None


def norm_text(s: str) -> str:
    """Normalization used ONLY for evaluation (offsets ignored)."""
    # collapse whitespace + trim
    s = re.sub(r"\s+", " ", s).strip()
    # case-insensitive matching helps with trivial casing differences (e.g., "The system" vs "the system")
    return s.lower()


def load_items_json(path: str) -> List[Dict[str, Any]]:
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


def spans_by_tag_set(item: Dict[str, Any]) -> Dict[str, set]:
    """Return {tag: set(span_text)} with normalization for eval."""
    m: Dict[str, set] = {t: set() for t in TAGS}
    for sp in (item.get("spans") or []):
        if not isinstance(sp, dict):
            continue
        tag = normalize_tag(sp.get("tag"))
        txt = sp.get("text")
        if tag and isinstance(txt, str) and txt.strip():
            m[tag].add(norm_text(txt))
    # Drop empty sets to keep output clean where needed
    return {k: v for k, v in m.items() if v}


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate_text_only(pred_path: str, gold_path: str) -> Dict[str, Any]:
    pred = index_by_id(load_items_json(pred_path))
    gold = index_by_id(load_items_json(gold_path))
    common = sorted(set(pred.keys()) & set(gold.keys()))
    if not common:
        raise RuntimeError("No common IDs between prediction and gold.")

    counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in TAGS}
    micro = {"tp": 0, "fp": 0, "fn": 0}

    # Optional: collect a small sample of mismatches for inspection
    samples: List[str] = []

    for rid in common:
        ps = spans_by_tag_set(pred[rid])
        gs = spans_by_tag_set(gold[rid])

        for t in TAGS:
            pset = ps.get(t, set())
            gset = gs.get(t, set())
            tp = len(pset & gset)
            fp = len(pset - gset)
            fn = len(gset - pset)

            counts[t]["tp"] += tp
            counts[t]["fp"] += fp
            counts[t]["fn"] += fn

            micro["tp"] += tp
            micro["fp"] += fp
            micro["fn"] += fn

        # add at most a few samples
        if len(samples) < 20:
            # pick one tag with mismatches if any
            for t in TAGS:
                pset = ps.get(t, set())
                gset = gs.get(t, set())
                extra = list(pset - gset)[:2]
                miss = list(gset - pset)[:2]
                if extra or miss:
                    samples.append(
                        f"ID={rid} | {t} | extra={extra} | missing={miss}"
                    )
                    break

    rows = []
    macro_sum = 0.0
    for t in TAGS:
        tp, fp, fn = counts[t]["tp"], counts[t]["fp"], counts[t]["fn"]
        p, r, f1 = prf(tp, fp, fn)
        macro_sum += f1
        rows.append({"tag": t, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1})

    macro_f1 = macro_sum / len(TAGS)
    micro_p, micro_r, micro_f1 = prf(micro["tp"], micro["fp"], micro["fn"])

    return {
        "rows": rows,
        "macro_f1": macro_f1,
        "micro": {**micro, "precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "common_ids": len(common),
        "samples": samples,
        "note": "Evaluation ignores offsets and compares only (tag, span_text) with whitespace+case normalization.",
    }


# -----------------------------
# GUI worker
# -----------------------------
@dataclass
class JobResult:
    json_path: str
    html_path: str


class Worker(QThread):
    done = pyqtSignal(object)
    error = pyqtSignal(str)
    info = pyqtSignal(str)

    def __init__(self, mode: str, dataset: str, ids: List[int], variant: str):
        super().__init__()
        self.mode = mode
        self.dataset = dataset
        self.ids = ids
        self.variant = variant

    def run(self):
        try:
            out_dir = Path(os.getenv("REQFLOW_RESULTS_DIR", "results"))
            if not out_dir.is_absolute():
                out_dir = (Path(__file__).resolve().parent / out_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            self.info.emit(f"Output dir: {out_dir}")

            if self.mode == "baseline":
                # Baseline = 1 agent, NO segmentation
                items = run_baseline(self.dataset, self.ids, variant=self.variant, with_offsets=True)
                out_json = out_dir / f"gui_baseline_{self.variant}.json"
            else:
                # Pipeline = agent-chain, may include Segmenter agent inside run_pipeline
                items = run_pipeline(self.dataset, self.ids, variant=self.variant, with_offsets=True)
                out_json = out_dir / f"gui_pipeline_{self.variant}.json"

            out_json.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
            out_html = out_json.with_suffix(".html")
            render_html(str(out_json), str(out_html), theme=os.getenv("REQFLOW_HTML_THEME", "dark"))

            self.done.emit(JobResult(str(out_json), str(out_html)))
        except Exception as e:
            self.error.emit(str(e))


# -----------------------------
# Main window
# -----------------------------
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ReqFlow A3 — Baseline vs Agent-Chain (Ollama)")
        self.resize(1280, 820)

        self.dataset_path = os.getenv("REQFLOW_DATASET", "data/dataset.csv")
        self.gold_path = os.getenv("REQFLOW_GOLD", "data/gold.json")

        self.last_json: Optional[str] = None
        self.last_html: Optional[str] = None

        # Controls
        self.mode_box = QComboBox()
        self.mode_box.addItems(["baseline", "pipeline"])

        self.variant_box = QComboBox()
        self.variant_box.addItems(["zero", "one", "few"])

        self.btn_load_dataset = QPushButton("📂 Load dataset")
        self.btn_pick_dataset = QPushButton("Change…")
        self.btn_pick_gold = QPushButton("Gold…")
        self.btn_run = QPushButton("▶ Run")
        self.btn_eval = QPushButton("✓ Evaluate last run")

        self.btn_open_pred = QPushButton("Open JSON")
        self.btn_open_html = QPushButton("Open HTML")
        self.btn_open_pred.setEnabled(False)
        self.btn_open_html.setEnabled(False)

        self.dataset_label = QLabel("Dataset: -")
        self.gold_label = QLabel("Gold: -")
        self.status = QLabel("Ready.")

        # IDs list
        self.ids_list = QListWidget()
        self.ids_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        # Viewer / log
        self.viewer = QWebEngineView()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 10))

        # Evaluation widgets
        self.eval_table = QTableWidget(0, 7)
        self.eval_table.setHorizontalHeaderLabels(["Tag", "P", "R", "F1", "TP", "FP", "FN"])
        self.eval_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 7):
            self.eval_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self.eval_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.eval_summary = QTextEdit()
        self.eval_summary.setReadOnly(True)
        self.eval_summary.setFont(QFont("Consolas", 10))

        # -----------------------------
        # Left panel (pretty grouping)
        # -----------------------------
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(10)

        gb_run = QGroupBox("Run configuration")
        gb_run_l = QVBoxLayout(gb_run)
        row1 = QWidget(); row1_l = QHBoxLayout(row1); row1_l.setContentsMargins(0,0,0,0)
        row1_l.addWidget(QLabel("Mode"))
        row1_l.addWidget(self.mode_box, 1)
        gb_run_l.addWidget(row1)

        row2 = QWidget(); row2_l = QHBoxLayout(row2); row2_l.setContentsMargins(0,0,0,0)
        row2_l.addWidget(QLabel("Variant"))
        row2_l.addWidget(self.variant_box, 1)
        gb_run_l.addWidget(row2)

        gb_files = QGroupBox("Files")
        gb_files_l = QVBoxLayout(gb_files)
        gb_files_l.addWidget(self.dataset_label)
        row_ds = QWidget(); row_ds_l = QHBoxLayout(row_ds); row_ds_l.setContentsMargins(0,0,0,0)
        row_ds_l.addWidget(self.btn_load_dataset)
        row_ds_l.addWidget(self.btn_pick_dataset)
        gb_files_l.addWidget(row_ds)
        gb_files_l.addWidget(self.gold_label)
        gb_files_l.addWidget(self.btn_pick_gold)

        gb_ids = QGroupBox("Requirement IDs (check to run)")
        gb_ids_l = QVBoxLayout(gb_ids)
        gb_ids_l.addWidget(self.ids_list)

        gb_actions = QGroupBox("Actions")
        gb_actions_l = QVBoxLayout(gb_actions)
        gb_actions_l.addWidget(self.btn_run)
        gb_actions_l.addWidget(self.btn_eval)
        row_open = QWidget(); row_open_l = QHBoxLayout(row_open); row_open_l.setContentsMargins(0,0,0,0)
        row_open_l.addWidget(self.btn_open_pred)
        row_open_l.addWidget(self.btn_open_html)
        gb_actions_l.addWidget(row_open)
        gb_actions_l.addWidget(self.status)

        lv.addWidget(gb_run)
        lv.addWidget(gb_files)
        lv.addWidget(gb_ids, 1)
        lv.addWidget(gb_actions)

        # -----------------------------
        # Tabs
        # -----------------------------
        tabs = QTabWidget()
        tabs.addTab(self.viewer, "HTML")
        tabs.addTab(self.log, "Log")

        eval_tab = QWidget()
        evl = QVBoxLayout(eval_tab)
        evl.setContentsMargins(10, 10, 10, 10)
        evl.setSpacing(10)
        evl.addWidget(self.eval_table, 2)
        evl.addWidget(QLabel("Notes / samples (first mismatches):"))
        evl.addWidget(self.eval_summary, 1)
        tabs.addTab(eval_tab, "Evaluation")

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(1, 1)

        root = QWidget()
        rl = QHBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(splitter)
        self.setCentralWidget(root)

        # signals
        self.btn_load_dataset.clicked.connect(self.load_dataset)
        self.btn_pick_dataset.clicked.connect(self.pick_dataset)
        self.btn_pick_gold.clicked.connect(self.pick_gold)
        self.btn_run.clicked.connect(self.run_job)

        self.btn_open_pred.clicked.connect(self.open_pred)
        self.btn_open_html.clicked.connect(self.open_html)
        self.btn_eval.clicked.connect(self.eval_last)

        # initial load
        self.refresh_labels()
        self.load_dataset()

    # -----------------------------
    # Helpers
    # -----------------------------
    def refresh_labels(self):
        self.dataset_label.setText(f"Dataset: {self.dataset_path}")
        self.gold_label.setText(f"Gold: {self.gold_path}")

    def append_log(self, msg: str):
        self.log.append(msg)
        self.log.ensureCursorVisible()

    def pick_dataset(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select dataset CSV", str(Path(self.dataset_path).parent), "CSV (*.csv)")
        if p:
            self.dataset_path = p
            self.refresh_labels()
            self.load_dataset()

    def pick_gold(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select gold JSON", str(Path(self.gold_path).parent), "JSON (*.json)")
        if p:
            self.gold_path = p
            self.refresh_labels()

    def load_dataset(self):
        try:
            df = pd.read_csv(self.dataset_path)
            if "text" not in df.columns and "text_en" in df.columns:
                df = df.rename(columns={"text_en": "text"})
            if "id" not in df.columns or "text" not in df.columns:
                raise ValueError("Dataset CSV must contain columns: id, text")

            self.ids_list.clear()
            for rid in df["id"].astype(int).tolist():
                it = QListWidgetItem(str(rid))
                it.setCheckState(Qt.CheckState.Unchecked)
                self.ids_list.addItem(it)

            self.status.setText(f"Loaded {len(df)} requirements.")
            self.append_log(f"Loaded dataset: {self.dataset_path} ({len(df)} rows)")
        except Exception as e:
            QMessageBox.critical(self, "Dataset error", str(e))

    def selected_ids(self) -> List[int]:
        ids: List[int] = []
        for i in range(self.ids_list.count()):
            it = self.ids_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                ids.append(int(it.text()))
        if not ids:
            # default: first 5
            for i in range(min(5, self.ids_list.count())):
                ids.append(int(self.ids_list.item(i).text()))
        return ids

    # -----------------------------
    # Run + callbacks
    # -----------------------------
    def run_job(self):
        mode = self.mode_box.currentText()
        variant = self.variant_box.currentText()
        ids = self.selected_ids()

        self.status.setText("Running…")
        self.btn_run.setEnabled(False)
        self.btn_eval.setEnabled(False)

        self.append_log(f"Run: mode={mode} variant={variant} ids={ids}")

        self.worker = Worker(mode, self.dataset_path, ids, variant)
        self.worker.done.connect(self.on_done)
        self.worker.error.connect(self.on_err)
        self.worker.info.connect(self.append_log)
        self.worker.start()

    def on_done(self, res: JobResult):
        self.last_json = res.json_path
        self.last_html = res.html_path
        self.btn_open_pred.setEnabled(True)
        self.btn_open_html.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_eval.setEnabled(True)
        self.status.setText("Done ✅")

        self.append_log(f"Wrote JSON: {res.json_path}")
        self.append_log(f"Wrote HTML: {res.html_path}")
        self.viewer.setUrl(QUrl.fromLocalFile(str(Path(res.html_path).resolve())))

    def on_err(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_eval.setEnabled(True)
        self.status.setText("Error ❌")
        QMessageBox.critical(self, "Error", msg)
        self.append_log(f"ERROR: {msg}")

    # -----------------------------
    # Open files
    # -----------------------------
    def open_pred(self):
        if self.last_json:
            if os.name == "nt":
                os.startfile(self.last_json)  # type: ignore[attr-defined]
            else:
                os.system(f'xdg-open "{self.last_json}"')

    def open_html(self):
        if self.last_html:
            if os.name == "nt":
                os.startfile(self.last_html)  # type: ignore[attr-defined]
            else:
                os.system(f'xdg-open "{self.last_html}"')

    # -----------------------------
    # Evaluation
    # -----------------------------
    def eval_last(self):
        if not self.last_json:
            QMessageBox.information(self, "Evaluation", "Run baseline/pipeline first to generate a prediction JSON.")
            return

        try:
            res = evaluate_text_only(self.last_json, self.gold_path)

            # Fill table
            rows = res["rows"]
            self.eval_table.setRowCount(len(rows) + 2)  # + macro + micro
            r = 0
            for row in rows:
                self._set_eval_row(r, row["tag"], row["precision"], row["recall"], row["f1"], row["tp"], row["fp"], row["fn"])
                r += 1

            # Macro
            self._set_eval_row(r, "MACRO_AVG", None, None, res["macro_f1"], None, None, None)
            r += 1
            micro = res["micro"]
            self._set_eval_row(r, "MICRO_AVG", micro["precision"], micro["recall"], micro["f1"], micro["tp"], micro["fp"], micro["fn"])

            # Summary
            lines = []
            lines.append(f"Common IDs evaluated: {res['common_ids']}")
            lines.append(f"Macro-F1: {res['macro_f1']:.4f}")
            lines.append(f"Micro-F1: {res['micro']['f1']:.4f}")
            lines.append("")
            lines.append(res["note"])
            lines.append("")
            if res["samples"]:
                lines.append("First mismatch samples:")
                lines.extend(res["samples"])
            self.eval_summary.setPlainText("\n".join(lines))

            self.status.setText("Evaluated ✅")
            self.append_log(f"Evaluation done on {res['common_ids']} common IDs. Macro-F1={res['macro_f1']:.4f}")
        except Exception as e:
            QMessageBox.critical(self, "Evaluation error", str(e))
            self.append_log(f"Evaluation ERROR: {e}")

    def _set_eval_row(self, row: int, tag: str,
                      p: Optional[float], r: Optional[float], f1: Optional[float],
                      tp: Optional[int], fp: Optional[int], fn: Optional[int]):
        def item(val: str):
            it = QTableWidgetItem(val)
            it.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            return it

        self.eval_table.setItem(row, 0, QTableWidgetItem(tag))

        self.eval_table.setItem(row, 1, item("" if p is None else f"{p:.4f}"))
        self.eval_table.setItem(row, 2, item("" if r is None else f"{r:.4f}"))
        self.eval_table.setItem(row, 3, item("" if f1 is None else f"{f1:.4f}"))

        self.eval_table.setItem(row, 4, item("" if tp is None else str(tp)))
        self.eval_table.setItem(row, 5, item("" if fp is None else str(fp)))
        self.eval_table.setItem(row, 6, item("" if fn is None else str(fn)))


def apply_fusion_theme(app: QApplication, mode: str):
    """
    mode: "dark" | "light"
    """
    app.setStyle("Fusion")
    if mode != "dark":
        return

    # Simple dark palette
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Base, QColor(22, 22, 22))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    pal.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(90, 140, 220))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(pal)


def main():
    load_dotenv()

    # Needed for some QtWebEngine setups
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    argv = sys.argv if sys.argv else ["reqflow"]
    app = QApplication(argv)

    # GUI theme (Qt widgets). HTML theme is separate (REQFLOW_HTML_THEME).
    ui_theme = os.getenv("REQFLOW_UI_THEME", "dark").strip().lower()
    apply_fusion_theme(app, ui_theme)

    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()