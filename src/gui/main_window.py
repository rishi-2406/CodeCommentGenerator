import os
import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QLabel, QPushButton, QStackedWidget, QSplitter, QTextEdit)
from PyQt6.QtCore import Qt, QSize, QObject, pyqtSignal
from PyQt6.QtGui import QFont, QIcon

from src.gui.generator_workspace import GeneratorWorkspace
from src.gui.insights_workspace import InsightsWorkspace
from src.gui.training_workspace import MLTrainingWorkspace
from src.gui.logs_workspace import LogsWorkspace
from src.gui.security_workspace import SecurityWorkspace

class StreamCapture(QObject):
    new_text = pyqtSignal(str)
    
    def write(self, text):
        self.new_text.emit(text)
        
    def flush(self):
        pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comment Gen Pro")
        self.setup_ui()
        self.setup_ui_connections()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 1. Top Header
        top_header = QWidget()
        top_header.setObjectName("TopHeader")
        th_layout = QHBoxLayout(top_header)
        th_layout.setContentsMargins(16, 8, 16, 8)
        
        title_lbl = QLabel("Comment Gen Pro")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 15px; color: #f1f5f9;")
        th_layout.addWidget(title_lbl)
        
        nav_text = QLabel("    File    Edit    View    Help")
        nav_text.setStyleSheet("color: #94a3b8; font-size: 13px; font-weight: 500; margin-left: 10px;")
        th_layout.addWidget(nav_text)
        
        th_layout.addStretch()
        
        self.btn_generate = QPushButton("Generate && Attach Comments")
        self.btn_generate.setProperty("class", "PrimaryButton")
        self.btn_generate.clicked.connect(self.trigger_generation)
        
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.setFixedSize(60, 32)
        
        th_layout.addWidget(self.btn_generate)
        th_layout.addWidget(QLabel(" | "))
        th_layout.addWidget(self.btn_settings)
        
        main_layout.addWidget(top_header)
        
        # 2. Body Splitter (Sidebar + content)
        body_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # -- Sidebar
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
        
        btn_new_proj = QPushButton("New Project")
        btn_new_proj.setProperty("class", "NewProjectButton")
        sb_layout.addWidget(btn_new_proj)
        
        body_splitter.addWidget(sidebar)
        
        # -- Main Content Area
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
        
        # Sub-splitter for Main Content vs Bottom Console
        content_splitter = QSplitter(Qt.Orientation.Vertical)
        content_splitter.addWidget(self.stacked_widget)
        
        # Bottom Console
        console_widget = QWidget()
        console_widget.setObjectName("PaneContainer")
        console_layout = QVBoxLayout(console_widget)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(0)
        
        console_header = QLabel("  Terminal Output")
        console_header.setStyleSheet("background: #0f172a; color: #cbd5e1; font-weight: 500; font-size: 12px; padding: 4px;")
        
        self.console_output = QTextEdit()
        self.console_output.setProperty("class", "CodeEditor")
        self.console_output.setReadOnly(True)
        self.console_output.setPlaceholderText("Live system output will appear here...")
        self.console_output.setStyleSheet("font-size: 11px; color: #98c379;")
        
        console_layout.addWidget(console_header)
        console_layout.addWidget(self.console_output)
        
        content_splitter.addWidget(console_widget)
        content_splitter.setSizes([600, 150])
        
        body_splitter.addWidget(content_splitter)
        
        main_layout.addWidget(body_splitter, stretch=1)
        
        # 3. Status Bar
        self.statusBar().showMessage("Status: Ready")
        
        # Intercept Print statements
        self.stdout_capture = StreamCapture()
        self.stdout_capture.new_text.connect(self.append_to_console)
        self.original_stdout = sys.stdout
        sys.stdout = self.stdout_capture
        sys.stderr = self.stdout_capture
        
        # Connections
        self.link_sidebar_buttons()
        
    def append_to_console(self, text):
        self.console_output.moveCursor(self.console_output.textCursor().MoveOperation.End)
        self.console_output.insertPlainText(text)
        self.console_output.verticalScrollBar().setValue(
            self.console_output.verticalScrollBar().maximum()
        )
        
    def closeEvent(self, event):
        sys.stdout = self.original_stdout
        sys.stderr = sys.__stderr__
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
        idx = self.stacked_widget.currentIndex()
        if idx == 0:
            self.statusBar().showMessage("Status: Running pipeline...")
            self.workspace_gen.trigger_generation()
            
    def setup_ui_connections(self):
        self.workspace_gen.generationStarted.connect(lambda: self.statusBar().showMessage("Status: Running pipeline..."))
        
        def on_gen_finished(status, code_unused, results_dict):
            self.statusBar().showMessage(f"Status: {status}")
            if results_dict:
                self.workspace_insights.populate_insights(results_dict)
                security_report = results_dict.get("security_report")
                self.workspace_security.populate(security_report)
                self.workspace_gen.update_unsafe_pct(security_report)
                if security_report:
                    unsafe_pct = round(100.0 - security_report.module_safe_pct, 1)
                    self.statusBar().showMessage(
                        f"Status: {status} | {security_report.module_safe_pct}% safe | "
                        f"{security_report.total_issues} security issues"
                    )
        
        self.workspace_gen.generationFinished.connect(on_gen_finished)
