"""
Generator Workspace UI Component
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSplitter, QTextEdit, QLineEdit, QRadioButton,
                             QFileDialog, QButtonGroup, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QSyntaxHighlighter
import sys
import os
import traceback

# Import the backend functionality
from src.main import run_pipeline
from src.logger import PipelineLogger
from src.gui.syntax_highlighter import PythonSyntaxHighlighter

class GeneratorWorker(QThread):
    finished = pyqtSignal(str, str, object) # success/error text, annotated_code, results_dict
    status = pyqtSignal(str)
    
    def __init__(self, code_text, is_ml, output_dir):
        super().__init__()
        self.code_text = code_text
        self.is_ml = is_ml
        self.output_dir = output_dir
        
    def run(self):
        try:
            self.status.emit("Setting up temporary files...")
            os.makedirs(self.output_dir, exist_ok=True)
            tmp_in = os.path.join(self.output_dir, "gui_tmp_in.py")
            with open(tmp_in, 'w', encoding='utf-8') as f:
                f.write(self.code_text)
                
            logger = PipelineLogger(input_file=tmp_in)
            
            model_selector = None
            if self.is_ml:
                self.status.emit("Loading ML Model...")
                # We assume models are already trained in the CWD
                try:
                    from src.ml.trainer import load_model_selector
                    model_selector = load_model_selector(output_dir="outputs")
                except Exception as e:
                    self.status.emit(f"Warning: Could not load ML models: {e}")
            
            self.status.emit("Running pipeline...")
            annotated, comments, mf, cg, attach_result, ir_module, analysis_report = \
                run_pipeline(tmp_in, logger, model_selector=model_selector)
                
            # Optionally save file
            out_file = os.path.join(self.output_dir, "gui_annotated.py")
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(annotated)
                
            self.status.emit("Saving logs...")
            logger.set_output_file(out_file)
            logger.save(os.path.join(self.output_dir, "logs")) # Save logs next to output
            
            results_dict = {
                "mf": mf,
                "cg": cg,
                "ir": ir_module,
                "analysis": analysis_report
            }
            self.finished.emit("Success", annotated, results_dict)
        except Exception as e:
            err = traceback.format_exc()
            self.finished.emit(f"Error: {e}\n\n{err}", "", None)

class GeneratorWorkspace(QWidget):
    generationStarted = pyqtSignal()
    generationFinished = pyqtSignal(str, str, object) # status_msg, annotated_code, results_dict
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_filepath = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Top Header (Engine Toggle + Output Dir)
        header = QWidget()
        header.setObjectName("WorkspaceHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        
        # Engine Toggle Group
        engine_group = QWidget()
        engine_layout = QHBoxLayout(engine_group)
        engine_layout.setContentsMargins(4, 4, 4, 4)
        engine_layout.setSpacing(8)
        
        self.btn_rule = QRadioButton("Rule-Based")
        self.btn_ml = QRadioButton("ML-Based CodeT5")
        self.btn_ml.setChecked(True)
        
        self.engine_btn_group = QButtonGroup(self)
        self.engine_btn_group.addButton(self.btn_rule, 1)
        self.engine_btn_group.addButton(self.btn_ml, 2)
        
        engine_layout.addWidget(self.btn_rule)
        engine_layout.addWidget(self.btn_ml)
        
        header_layout.addWidget(engine_group)
        header_layout.addStretch()
        
        # Output Dir
        dir_layout = QHBoxLayout()
        dir_label = QLabel("Output Dir:")
        dir_label.setStyleSheet("color: #cbd5e1; font-weight: 500; font-size: 14px;")
        
        self.out_dir_input = QLineEdit("./out/annotated")
        self.out_dir_input.setProperty("class", "DirInput")
        self.out_dir_input.setMinimumWidth(250)
        
        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(self.out_dir_input)
        header_layout.addLayout(dir_layout)
        
        layout.addWidget(header)
        
        # Splitter (Input left, Output right)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Input Pane
        input_pane = QWidget()
        input_pane.setObjectName("PaneContainer")
        input_layout = QVBoxLayout(input_pane)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(0)
        
        input_header = QWidget()
        input_header.setObjectName("EditorPaneHeader")
        ih_layout = QHBoxLayout(input_header)
        ih_layout.setContentsMargins(16, 8, 16, 8)
        ih_title = QLabel("📄 Input Python Code")
        self.btn_pick_file = QPushButton("📂 Pick File")
        self.btn_pick_file.setProperty("class", "SecondaryButton")
        self.btn_pick_file.clicked.connect(self.pick_file)
        ih_layout.addWidget(ih_title)
        ih_layout.addStretch()
        ih_layout.addWidget(self.btn_pick_file)
        
        self.input_editor = QTextEdit()
        self.input_editor.setProperty("class", "CodeEditor")
        self.input_editor.setPlaceholderText("Paste Python code here, or Pick File...")
        self.input_highlighter = PythonSyntaxHighlighter(self.input_editor.document())
        
        input_layout.addWidget(input_header)
        input_layout.addWidget(self.input_editor)
        
        # Output Pane
        output_pane = QWidget()
        output_pane.setObjectName("PaneContainer")
        output_layout = QVBoxLayout(output_pane)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(0)
        
        output_header = QWidget()
        output_header.setObjectName("EditorPaneHeader")
        oh_layout = QHBoxLayout(output_header)
        oh_layout.setContentsMargins(16, 8, 16, 8)
        oh_title = QLabel("🔍 Annotated Output")
        
        status_lbl = QLabel("🟢 New Comments")
        status_lbl.setStyleSheet("color: #4ade80; font-size: 12px;")
        oh_layout.addWidget(oh_title)
        oh_layout.addStretch()
        oh_layout.addWidget(status_lbl)
        
        self.output_editor = QTextEdit()
        self.output_editor.setProperty("class", "DiffEditor")
        self.output_editor.setReadOnly(True)
        self.output_editor.setPlaceholderText("Annotated code will appear here...")
        self.output_highlighter = PythonSyntaxHighlighter(self.output_editor.document())
        
        output_layout.addWidget(output_header)
        output_layout.addWidget(self.output_editor)
        
        self.splitter.addWidget(input_pane)
        self.splitter.addWidget(output_pane)
        
        # Set a slight grey divider color
        self.splitter.setStyleSheet("QSplitter::handle { background-color: #1e293b; }")
        
        layout.addWidget(self.splitter, stretch=1)
        
    def pick_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Python File", "", "Python Files (*.py);;All Files (*)"
        )
        if filepath:
            self.current_filepath = filepath
            with open(filepath, 'r', encoding='utf-8') as f:
                self.input_editor.setPlainText(f.read())

    def trigger_generation(self):
        code = self.input_editor.toPlainText().strip()
        if not code:
            QMessageBox.warning(self, "No Code", "Please pick a file or paste some code.")
            return
            
        is_ml = self.btn_ml.isChecked()
        out_dir = self.out_dir_input.text()
        
        self.worker = GeneratorWorker(code, is_ml, out_dir)
        self.worker.status.connect(lambda msg: self.generationStarted.emit()) # proxy
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.start()

    def on_generation_finished(self, status, code, results_dict):
        if status == "Success":
            self.output_editor.setPlainText(code)
            self.generationFinished.emit("Generation Completed", "", results_dict)
        else:
            self.output_editor.setPlainText(status) # Dump error
            self.generationFinished.emit("Generation Failed", "", None)

    def update_annotated_code(self, code):
        self.output_editor.setPlainText(code)
