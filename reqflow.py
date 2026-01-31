from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSplitter,
    QTabWidget, QVBoxLayout, QWidget, QTextEdit
)
from PyQt6.QtWebEngineWidgets import QWebEngineView

from src.env import load_dotenv
from src.baseline import run_baseline
from src.pipeline import run_pipeline
from src.render import main as render_html

@dataclass
class JobResult:
    json_path: str
    html_path: str

class Worker(QThread):
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, mode: str, dataset: str, ids: List[int], variant: str):
        super().__init__()
        self.mode = mode
        self.dataset = dataset
        self.ids = ids
        self.variant = variant

    def run(self):
        try:
            out_dir = Path(os.getenv("REQFLOW_RESULTS_DIR","results"))
            if not out_dir.is_absolute():
                out_dir = (Path(__file__).resolve().parent / out_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            if self.mode == "baseline":
                items = run_baseline(self.dataset, self.ids, variant=self.variant, with_offsets=True)
                out_json = out_dir / f"gui_baseline_{self.variant}.json"
            else:
                items = run_pipeline(self.dataset, self.ids, variant=self.variant, with_offsets=True)
                out_json = out_dir / f"gui_pipeline_{self.variant}.json"

            out_json.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
            out_html = out_json.with_suffix(".html")
            render_html(str(out_json), str(out_html), theme=os.getenv("REQFLOW_HTML_THEME","dark"))

            self.done.emit(JobResult(str(out_json), str(out_html)))
        except Exception as e:
            self.error.emit(str(e))

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ReqFlow A3 — Baseline vs Agent-Chain")
        self.resize(1200, 800)

        self.dataset_path = os.getenv("REQFLOW_DATASET","data/dataset.csv")
        self.gold_path = os.getenv("REQFLOW_GOLD","data/gold.json")

        self.ids_list = QListWidget()
        self.ids_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["baseline","pipeline"])

        self.variant_box = QComboBox()
        self.variant_box.addItems(["zero","one","few"])

        self.btn_load = QPushButton("Load dataset")
        self.btn_run = QPushButton("Run")
        self.btn_open_pred = QPushButton("Open prediction JSON")
        self.btn_open_html = QPushButton("Open HTML")
        self.btn_open_pred.setEnabled(False)
        self.btn_open_html.setEnabled(False)

        self.status = QLabel("Ready.")
        self.viewer = QWebEngineView()
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # layout
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Mode"))
        lv.addWidget(self.mode_box)
        lv.addWidget(QLabel("Variant"))
        lv.addWidget(self.variant_box)
        lv.addWidget(self.btn_load)
        lv.addWidget(QLabel("Requirement IDs"))
        lv.addWidget(self.ids_list, 1)
        lv.addWidget(self.btn_run)
        lv.addWidget(self.btn_open_pred)
        lv.addWidget(self.btn_open_html)
        lv.addWidget(self.status)

        tabs = QTabWidget()
        tabs.addTab(self.viewer, "HTML")
        tabs.addTab(self.log, "Log")

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(1, 1)

        root = QWidget()
        rl = QHBoxLayout(root)
        rl.addWidget(splitter)
        self.setCentralWidget(root)

        self.btn_load.clicked.connect(self.load_dataset)
        self.btn_run.clicked.connect(self.run_job)
        self.btn_open_pred.clicked.connect(self.open_pred)
        self.btn_open_html.clicked.connect(self.open_html)

        self.last_json: Optional[str] = None
        self.last_html: Optional[str] = None

        self.load_dataset()

    def load_dataset(self):
        try:
            df = pd.read_csv(self.dataset_path)
            if "text" not in df.columns and "text_en" in df.columns:
                df = df.rename(columns={"text_en":"text"})
            self.ids_list.clear()
            for rid in df["id"].astype(int).tolist():
                it = QListWidgetItem(str(rid))
                it.setCheckState(Qt.CheckState.Unchecked)
                self.ids_list.addItem(it)
            self.status.setText(f"Loaded {len(df)} requirements from {self.dataset_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def selected_ids(self) -> List[int]:
        ids = []
        for i in range(self.ids_list.count()):
            it = self.ids_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                ids.append(int(it.text()))
        if not ids:
            # default: first 5
            for i in range(min(5, self.ids_list.count())):
                ids.append(int(self.ids_list.item(i).text()))
        return ids

    def run_job(self):
        mode = self.mode_box.currentText()
        variant = self.variant_box.currentText()
        ids = self.selected_ids()

        self.status.setText("Running...")
        self.log.append(f"Run: mode={mode}, variant={variant}, ids={ids}")

        self.worker = Worker(mode, self.dataset_path, ids, variant)
        self.worker.done.connect(self.on_done)
        self.worker.error.connect(self.on_err)
        self.worker.start()

    def on_done(self, res: JobResult):
        self.last_json = res.json_path
        self.last_html = res.html_path
        self.btn_open_pred.setEnabled(True)
        self.btn_open_html.setEnabled(True)
        self.status.setText("Done.")
        self.log.append(f"Wrote: {res.json_path}")
        self.log.append(f"Wrote: {res.html_path}")
        self.viewer.setUrl(QUrl.fromLocalFile(str(Path(res.html_path).resolve())))

    def on_err(self, msg: str):
        self.status.setText("Error.")
        QMessageBox.critical(self, "Error", msg)

    def open_pred(self):
        if self.last_json:
            os.startfile(self.last_json) if os.name == "nt" else os.system(f'xdg-open "{self.last_json}"')

    def open_html(self):
        if self.last_html:
            os.startfile(self.last_html) if os.name == "nt" else os.system(f'xdg-open "{self.last_html}"')

def main():
    load_dotenv()
    argv = sys.argv if sys.argv else ["reqflow"]
    app = QApplication(argv)
    w = Main()
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
