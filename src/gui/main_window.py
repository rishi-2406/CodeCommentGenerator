import os
import sys
import re
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QLabel, QPushButton, QStackedWidget, QSplitter, QTextEdit)
from PyQt6.QtCore import Qt, QSize, QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QTextCursor, QTextCharFormat, QColor

from src.gui.generator_workspace import GeneratorWorkspace
from src.gui.insights_workspace import InsightsWorkspace
from src.gui.training_workspace import MLTrainingWorkspace
from src.gui.logs_workspace import LogsWorkspace
from src.gui.security_workspace import SecurityWorkspace
from src.gui.widgets import SpinningButton, ToastWidget


class StreamCapture(QObject):
    new_text = pyqtSignal(str, bool)

    def __init__(self, is_stderr=False):
        super().__init__()
        self._is_stderr = is_stderr

    def write(self, text):
        self.new_text.emit(text, self._is_stderr)

    def flush(self):
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comment Gen Pro")
        self._status_reset_timer = QTimer(self)
        self._status_reset_timer.setSingleShot(True)
        self._status_reset_timer.timeout.connect(self._reset_status_style)
        self.setup_ui()
        self.setup_ui_connections()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self._toast_host = central_widget

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_header = QWidget()
        top_header.setObjectName("TopHeader")
        th_layout = QHBoxLayout(top_header)
        th_layout.setContentsMargins(16, 8, 16, 8)

        title_lbl = QLabel("Comment Gen Pro")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 15px; color: #1a1a2e;")
        th_layout.addWidget(title_lbl)

        th_layout.addStretch()

        self.btn_generate = SpinningButton("Generate && Attach Comments")
        self.btn_generate.setProperty("class", "PrimaryButton")
        self.btn_generate.clicked.connect(self.trigger_generation)

        th_layout.addWidget(self.btn_generate)

        main_layout.addWidget(top_header)

        body_splitter = QSplitter(Qt.Orientation.Horizontal)

        sidebar = QWidget()
        sidebar.setObjectName("SidebarWidget")
        sidebar.setMinimumWidth(200)
        sidebar.setMaximumWidth(250)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(16, 16, 16, 16)
        sb_layout.setSpacing(8)

        workspace_lbl = QLabel("Workspace")
        workspace_lbl.setObjectName("SidebarTitle")
        local_env_lbl = QLabel("Local Env")
        local_env_lbl.setObjectName("SidebarSubtitle")

        sb_layout.addWidget(workspace_lbl)
        sb_layout.addWidget(local_env_lbl)
        sb_layout.addSpacing(12)

        self.btn_nav_gen = QPushButton("Generator")
        self.btn_nav_gen.setProperty("class", "SidebarButton")
        self.btn_nav_gen.setProperty("active", "true")

        self.btn_nav_insights = QPushButton("Insights")
        self.btn_nav_insights.setProperty("class", "SidebarButton")

        self.btn_nav_ml = QPushButton("ML Training")
        self.btn_nav_ml.setProperty("class", "SidebarButton")

        self.btn_nav_security = QPushButton("Security")
        self.btn_nav_security.setProperty("class", "SidebarButton")

        self.btn_nav_logs = QPushButton("Logs")
        self.btn_nav_logs.setProperty("class", "SidebarButton")

        sb_layout.addWidget(self.btn_nav_gen)
        sb_layout.addWidget(self.btn_nav_insights)
        sb_layout.addWidget(self.btn_nav_ml)
        sb_layout.addWidget(self.btn_nav_security)
        sb_layout.addWidget(self.btn_nav_logs)

        sb_layout.addStretch()

        body_splitter.addWidget(sidebar)

        self.stacked_widget = QStackedWidget()

        self.workspace_gen = GeneratorWorkspace()
        self.stacked_widget.addWidget(self.workspace_gen)

        self.workspace_insights = InsightsWorkspace()
        self.stacked_widget.addWidget(self.workspace_insights)

        self.workspace_ml = MLTrainingWorkspace()
        self.stacked_widget.addWidget(self.workspace_ml)

        self.workspace_security = SecurityWorkspace()
        self.stacked_widget.addWidget(self.workspace_security)

        self.workspace_logs = LogsWorkspace()
        self.stacked_widget.addWidget(self.workspace_logs)

        content_splitter = QSplitter(Qt.Orientation.Vertical)
        content_splitter.addWidget(self.stacked_widget)

        console_widget = QWidget()
        console_widget.setObjectName("PaneContainer")
        console_layout = QVBoxLayout(console_widget)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(0)

        console_header = QLabel("  Terminal Output")
        console_header.setStyleSheet("background: #f8f9fa; color: #495057; font-weight: 500; font-size: 12px; padding: 6px; border-bottom: 1px solid #e0e4e8;")

        self.console_output = QTextEdit()
        self.console_output.setProperty("class", "CodeEditor")
        self.console_output.setReadOnly(True)
        self.console_output.setPlaceholderText("Live system output will appear here...")
        self.console_output.setStyleSheet("font-size: 11px; color: #1a1a2e; background: #ffffff;")

        console_layout.addWidget(console_header)
        console_layout.addWidget(self.console_output)

        content_splitter.addWidget(console_widget)
        content_splitter.setSizes([600, 150])

        body_splitter.addWidget(content_splitter)

        main_layout.addWidget(body_splitter, stretch=1)

        self.statusBar().showMessage("Status: Ready")
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("padding: 0px 8px; color: #6c757d;")
        self.statusBar().addPermanentWidget(self._status_label)

        self.stdout_capture = StreamCapture(is_stderr=False)
        self.stdout_capture.new_text.connect(self.append_to_console)
        self.stderr_capture = StreamCapture(is_stderr=True)
        self.stderr_capture.new_text.connect(self.append_to_console)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture

        self.toast = ToastWidget(self._toast_host)
        self.toast.hide()

        self.link_sidebar_buttons()

    def append_to_console(self, text, is_stderr=False):
        cursor = self.console_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        if is_stderr:
            fmt.setForeground(QColor("#dc2626"))
        elif re.search(r'\b(error|fail|exception|traceback)\b', text, re.IGNORECASE):
            fmt.setForeground(QColor("#dc2626"))
        elif re.search(r'\b(success|completed|saved|done|finished)\b', text, re.IGNORECASE):
            fmt.setForeground(QColor("#16a34a"))
        else:
            fmt.setForeground(QColor("#1a1a2e"))

        cursor.setCharFormat(fmt)
        cursor.insertText(text)

        self.console_output.verticalScrollBar().setValue(
            self.console_output.verticalScrollBar().maximum()
        )

    def _set_status(self, message, color="#6c757d"):
        self._status_label.setText(message)
        self._status_label.setStyleSheet(
            f"padding: 0px 8px; color: {color}; font-weight: 600;"
        )
        self.statusBar().showMessage("")

    def _reset_status_style(self):
        self._set_status("Ready", "#6c757d")

    def closeEvent(self, event):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        super().closeEvent(event)

    def link_sidebar_buttons(self):
        self.nav_buttons = [
            self.btn_nav_gen,
            self.btn_nav_insights,
            self.btn_nav_ml,
            self.btn_nav_security,
            self.btn_nav_logs,
        ]

        def set_active(idx, active_btn):
            self.stacked_widget.setCurrentIndex(idx)
            for btn in self.nav_buttons:
                btn.setProperty("active", "true" if btn == active_btn else "false")
                btn.style().unpolish(btn)
                btn.style().polish(btn)

        self.btn_nav_gen.clicked.connect(lambda: set_active(0, self.btn_nav_gen))
        self.btn_nav_insights.clicked.connect(lambda: set_active(1, self.btn_nav_insights))
        self.btn_nav_ml.clicked.connect(lambda: set_active(2, self.btn_nav_ml))
        self.btn_nav_security.clicked.connect(lambda: set_active(3, self.btn_nav_security))
        self.btn_nav_logs.clicked.connect(lambda: set_active(4, self.btn_nav_logs))

    def trigger_generation(self):
        if self.btn_generate.is_spinning():
            return

        idx = self.stacked_widget.currentIndex()
        if idx != 0:
            self.stacked_widget.setCurrentIndex(0)
            for btn in self.nav_buttons:
                btn.setProperty("active", "true" if btn == self.btn_nav_gen else "false")
                btn.style().unpolish(btn)
                btn.style().polish(btn)

        self._set_status("Generating...", "#2563eb")
        self.btn_generate.start_spinning("Generating...")
        self.workspace_gen.trigger_generation()

    def _on_generation_started(self):
        self._set_status("Generating...", "#2563eb")

    def _on_generation_finished(self, status, code_unused, results_dict):
        self.btn_generate.stop_spinning()
        self._status_reset_timer.start(8000)

        if "Completed" in status or status == "Success":
            safe_msg = ""
            if results_dict:
                self.workspace_insights.populate_insights(results_dict)
                security_report = results_dict.get("security_report")
                self.workspace_security.populate(security_report)
                self.workspace_gen.update_unsafe_pct(security_report)
                if security_report:
                    safe_msg = f" | {security_report.module_safe_pct}% safe | {security_report.total_issues} issues"
            self._set_status(f"\u2713 Generation Completed{safe_msg}", "#16a34a")
            self.toast.show_toast("Comments generated successfully", kind="success")
        else:
            self._set_status(f"\u2717 Generation Failed", "#dc2626")
            self.toast.show_toast("Generation failed — check output for details", kind="error")

    def setup_ui_connections(self):
        self.workspace_gen.generationStarted.connect(self._on_generation_started)
        self.workspace_gen.generationFinished.connect(self._on_generation_finished)
