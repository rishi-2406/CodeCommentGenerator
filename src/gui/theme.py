# -*- coding: utf-8 -*-
"""
PyQt6 theme module — Professional Light Theme.
"""

MAIN_STYLESHEET = """
QMainWindow {
    background-color: #f8f9fa;
    color: #1a1a2e;
}

QWidget {
    color: #1a1a2e;
    font-family: 'Inter', sans-serif;
}

#SidebarWidget {
    background-color: #ffffff;
    border-right: 1px solid #e0e4e8;
}

#SidebarTitle {
    color: #1a1a2e;
    font-size: 16px;
    font-weight: bold;
}

#SidebarSubtitle {
    color: #6c757d;
    font-size: 14px;
}

QPushButton.SidebarButton {
    background-color: transparent;
    color: #495057;
    text-align: left;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    border: none;
}

QPushButton.SidebarButton:hover {
    background-color: #f1f3f5;
}

QPushButton.SidebarButton[active="true"] {
    background-color: #e8f0fe;
    color: #2563eb;
    font-weight: 600;
}

#TopHeader {
    background-color: #ffffff;
    border-bottom: 1px solid #e0e4e8;
}

QPushButton.PrimaryButton {
    background-color: #2563eb;
    color: #ffffff;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 13px;
    border: none;
}

QPushButton.PrimaryButton:hover {
    background-color: #1d4ed8;
}

QPushButton.PrimaryButton:pressed {
    background-color: #1e40af;
}

QPushButton.PrimaryButton:disabled {
    background-color: #93c5fd;
    color: #ffffff;
}

#WorkspaceHeader {
    background-color: #ffffff;
    border-bottom: 1px solid #e0e4e8;
}

QLineEdit.DirInput {
    background-color: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 6px;
    padding: 6px 12px;
    color: #1a1a2e;
    font-size: 14px;
}

QLineEdit.DirInput:focus {
    border: 2px solid #2563eb;
    padding: 5px 11px;
}

QTextEdit.CodeEditor {
    background-color: #ffffff;
    color: #1a1a2e;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 13px;
    border: none;
    line-height: 1.5;
}

QTextEdit.DiffEditor {
    background-color: #ffffff;
    color: #1a1a2e;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 13px;
    border: none;
    line-height: 1.5;
}

#EditorPaneHeader {
    background-color: #f8f9fa;
    border-bottom: 1px solid #e0e4e8;
}

#EditorPaneHeader QLabel {
    color: #495057;
    font-size: 14px;
    font-weight: 500;
}

QPushButton.SecondaryButton {
    background-color: #ffffff;
    color: #495057;
    border: 1px solid #d0d5dd;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
}

QPushButton.SecondaryButton:hover {
    background-color: #f1f3f5;
    border-color: #b0b7c3;
}

#PaneContainer {
    background-color: #ffffff;
    border: 1px solid #e0e4e8;
}

QStatusBar {
    background-color: #ffffff;
    color: #6c757d;
    border-top: 1px solid #e0e4e8;
    font-size: 13px;
}

QStatusBar::item {
    border: none;
}

QSplitter::handle {
    background-color: #e0e4e8;
}

QRadioButton {
    color: #495057;
    font-size: 13px;
    spacing: 6px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid #b0b7c3;
    background-color: #ffffff;
}

QRadioButton::indicator:checked {
    border-color: #2563eb;
    background-color: #2563eb;
}

QTabWidget::pane {
    border: 1px solid #e0e4e8;
    background: #ffffff;
    border-radius: 6px;
}

QTabBar::tab {
    background: #f1f3f5;
    color: #6c757d;
    padding: 8px 18px;
    border: 1px solid #e0e4e8;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 500;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #ffffff;
    color: #2563eb;
    border-bottom-color: #ffffff;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background: #e8f0fe;
    color: #2563eb;
}

QTreeWidget {
    background-color: #ffffff;
    color: #1a1a2e;
    border: 1px solid #e0e4e8;
    font-size: 13px;
    border-radius: 4px;
}

QTreeWidget::item {
    padding: 3px;
}

QTreeWidget::item:selected {
    background-color: #e8f0fe;
    color: #2563eb;
}

QTreeWidget::item:hover {
    background-color: #f1f3f5;
}

QHeaderView::section {
    background-color: #f8f9fa;
    color: #495057;
    padding: 6px 8px;
    border: 1px solid #e0e4e8;
    font-weight: 600;
    font-size: 12px;
}

QTableWidget {
    background-color: #ffffff;
    color: #1a1a2e;
    gridline-color: #e0e4e8;
    border: 1px solid #e0e4e8;
    font-size: 13px;
    border-radius: 4px;
}

QTableWidget::item:selected {
    background-color: #e8f0fe;
    color: #2563eb;
}

QTableCornerButton::section {
    background-color: #f8f9fa;
    border: 1px solid #e0e4e8;
}

QProgressBar {
    background-color: #e0e4e8;
    border: none;
    border-radius: 2px;
    height: 4px;
    max-height: 4px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 2px;
}

QScrollBar:vertical {
    background: #f1f3f5;
    width: 10px;
    border: none;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #c4c9d1;
    min-height: 30px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #a0a7b1;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #f1f3f5;
    height: 10px;
    border: none;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #c4c9d1;
    min-width: 30px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #a0a7b1;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QToolTip {
    background-color: #1a1a2e;
    color: #f1f3f5;
    border: 1px solid #495057;
    padding: 6px 10px;
    font-size: 12px;
    border-radius: 4px;
}
"""
