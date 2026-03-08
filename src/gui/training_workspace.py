import sys
import traceback
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTextEdit, QScrollArea, QFrame)
from PyQt6.QtCore import pyqtSignal, QThread
import json

from src.ml.trainer import train_and_evaluate

class TrainerWorker(QThread):
    status = pyqtSignal(str)
    finished = pyqtSignal(object) # dict of reports

    def run(self):
        try:
            self.status.emit("Starting ML training pipeline. Check Terminal Output below...")
            # Run the trainer. With verbose=True it will print to the global Terminal Output.
            result = train_and_evaluate(
                output_dir="outputs/gui_models",
                verbose=True
            )
            self.status.emit("Training completed successfully!")
            self.finished.emit(result)
        except Exception as e:
            err = traceback.format_exc()
            self.status.emit(f"Error during training:\n{err}")
            self.finished.emit({"error": str(e), "trace": err})


class MLTrainingWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        header_layout = QHBoxLayout()
        title = QLabel("🧠 ML Training Dashboard")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        header_layout.addWidget(title)
        
        self.btn_train = QPushButton("🚀 Start / Retrain Models")
        self.btn_train.setProperty("class", "PrimaryButton")
        self.btn_train.setMinimumHeight(35)
        self.btn_train.clicked.connect(self.start_training)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_train)
        
        layout.addLayout(header_layout)
        
        # Status Label
        self.status_lbl = QLabel("Ready to train models.")
        self.status_lbl.setStyleSheet("color: #94a3b8;")
        layout.addWidget(self.status_lbl)
        
        # Results area
        self.results_area = QTextEdit()
        self.results_area.setProperty("class", "CodeEditor")
        self.results_area.setReadOnly(True)
        self.results_area.setPlaceholderText("Training reports will appear here...")
        
        layout.addWidget(self.results_area)
        
    def start_training(self):
        self.btn_train.setEnabled(False)
        self.status_lbl.setText("Running training pipeline...")
        self.results_area.clear()
        
        self.worker = TrainerWorker()
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.start()
        
    def update_status(self, msg):
        self.status_lbl.setText(msg)
        self.results_area.append(msg)
        
    def on_training_finished(self, result):
        self.btn_train.setEnabled(True)
        if "error" in result:
            return
            
        tr = result.get("training_report", {})
        ev = result.get("eval_report", {})
        
        report_text = f"""
======================================
  TRAINING COMPLETED
======================================
Dataset Total Samples  : {tr.get('dataset_total', 0)}
Training Samples (Seed): {tr.get('train_size', 0)}
Testing Samples (Seed) : {tr.get('test_size', 0)}

CodeT5 Finetuning Corpus : {tr.get('corpus_total', 0)} pairs

EVALUATION METRICS
--------------------------------------
"""
        summary = ev.get('summary', {})
        report_text += f"Best BLEU-4 Score : {summary.get('best_bleu4', 0):.4f}\n"
        report_text += f"Best ROUGE-L Score: {summary.get('best_rouge_l', 0):.4f}\n"
        report_text += f"Exact Match Rate  : {summary.get('best_exact_match', 0):.4f}\n\n"
        
        report_text += "----- FULL JSON REPORTS -----\n"
        report_text += json.dumps(result, indent=2)
        
        self.results_area.setPlainText(report_text)
