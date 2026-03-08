# -*- coding: utf-8 -*-
"""
PyQt6 theme module mimicking the Stitch Dark Mode Tailwind design.
"""

MAIN_STYLESHEET = """
QMainWindow {
    background-color: #111921;
    color: #f1f5f9;
}

QWidget {
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

/* SIDEBAR */
#SidebarWidget {
    background-color: #18232c;
    border-right: 1px solid #1e293b;
}

#SidebarTitle {
    color: #f1f5f9;
    font-size: 16px;
    font-weight: bold;
}

#SidebarSubtitle {
    color: #94a3b8;
    font-size: 14px;
}

QPushButton.SidebarButton {
    background-color: transparent;
    color: #cbd5e1;
    text-align: left;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    border: none;
}

QPushButton.SidebarButton:hover {
    background-color: rgba(30, 41, 59, 0.5); /* #1e293b with opacity */
}

QPushButton.SidebarButton[active="true"] {
    background-color: rgba(25, 127, 230, 0.2); /* #197fe6 with opacity */
    color: #197fe6;
}

QPushButton.NewProjectButton {
    background-color: rgba(25, 127, 230, 0.2);
    color: #197fe6;
    border-radius: 6px;
    padding: 8px;
    font-size: 14px;
    border: none;
}
QPushButton.NewProjectButton:hover {
    background-color: rgba(25, 127, 230, 0.3);
}

/* TOP HEADER */
#TopHeader {
    background-color: #18232c;
    border-bottom: 1px solid #1e293b;
}

QPushButton.PrimaryButton {
    background-color: #197fe6;
    color: white;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: bold;
    font-size: 13px;
    border: none;
}

QPushButton.PrimaryButton:hover {
    background-color: #156ac0;
}

/* TAB BAR / SETTINGS STRIP */
#WorkspaceHeader {
    background-color: #18232c;
    border-bottom: 1px solid #1e293b;
}

QLineEdit.DirInput {
    background-color: #111921;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 10px;
    color: #f1f5f9;
    font-size: 14px;
}
QLineEdit.DirInput:focus {
    border: 1px solid #197fe6;
}

/* TEXT EDITORS (CODE) */
QTextEdit.CodeEditor {
    background-color: transparent;
    color: #cbd5e1;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 13px;
    border: none;
    line-height: 1.5;
}

QTextEdit.DiffEditor {
    background-color: transparent;
    color: #cbd5e1;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 13px;
    border: none;
    line-height: 1.5;
}

/* PANES */
#EditorPaneHeader {
    background-color: #0f172a;
    border-bottom: 1px solid #1e293b;
}
#EditorPaneHeader QLabel {
    color: #cbd5e1;
    font-size: 14px;
    font-weight: 500;
}

QPushButton.SecondaryButton {
    background-color: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}
QPushButton.SecondaryButton:hover {
    background-color: #1e293b;
}

#PaneContainer {
    background-color: #1e293b;
}

/* STATUS BAR */
QStatusBar {
    background-color: #18232c;
    color: #94a3b8;
    border-top: 1px solid #1e293b;
}
QStatusBar::item {
    border: none;
}
"""
