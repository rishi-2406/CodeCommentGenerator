import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTextEdit, QSplitter, QTreeWidget, QTreeWidgetItem)
from PyQt6.QtCore import Qt
from src.gui.syntax_highlighter import JsonSyntaxHighlighter

class LogsWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # We will parse out/annotated/logs and outputs/
        self.output_dirs = {
            "Pipeline Logs": "out/annotated/logs",
            "Model Training Outputs": "outputs",
            "Model Training Logs": "outputs/logs"
        }
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        header_layout = QHBoxLayout()
        title = QLabel("📋 Logs & Reports Viewer")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        self.btn_load_logs = QPushButton("🔄 Refresh Files")
        self.btn_load_logs.setProperty("class", "SecondaryButton")
        self.btn_load_logs.clicked.connect(self.load_log_list)
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_load_logs)
        layout.addLayout(header_layout)
        
        # Splitter for files list and viewer
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setStyleSheet("""
            QTreeWidget {
                background-color: #111921;
                border: 1px solid #1e293b;
                color: #cbd5e1;
                font-size: 13px;
                padding: 4px;
            }
            QTreeWidget::item {
                padding: 4px;
            }
            QTreeWidget::item:selected {
                background-color: #197fe6;
                color: white;
            }
        """)
        self.file_tree.itemClicked.connect(self.on_file_selected)
        
        self.log_viewer = QTextEdit()
        self.log_viewer.setProperty("class", "CodeEditor")
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setPlaceholderText("Select a file from the tree to view...")
        self.hl_logs = JsonSyntaxHighlighter(self.log_viewer.document())
        
        self.splitter.addWidget(self.file_tree)
        self.splitter.addWidget(self.log_viewer)
        self.splitter.setSizes([250, 600])
        
        layout.addWidget(self.splitter, stretch=1)
        self.load_log_list()
        
    def load_log_list(self):
        self.file_tree.clear() # Reset
        
        for category, search_dir in self.output_dirs.items():
            if not os.path.exists(search_dir):
                continue
                
            # Create category top level item
            cat_item = QTreeWidgetItem([category])
            cat_item.setExpanded(True)
            self.file_tree.addTopLevelItem(cat_item)
            
            # Fetch valid files
            files = []
            for f in os.listdir(search_dir):
                if f.endswith(".log") or f.endswith(".json") or f.endswith(".txt") or f.endswith(".csv"):
                    abs_path = os.path.abspath(os.path.join(search_dir, f))
                    if os.path.isfile(abs_path):
                        files.append((f, abs_path))
            
            # Sort chronologically by modified time
            files.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
            
            for f_name, f_path in files:
                child = QTreeWidgetItem([f_name])
                child.setData(0, Qt.ItemDataRole.UserRole, f_path) # Store path in data
                cat_item.addChild(child)
                        
    def on_file_selected(self, item, column):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.log_viewer.setPlainText(content)
            except Exception as e:
                self.log_viewer.setPlainText(f"Error reading file: {e}")
