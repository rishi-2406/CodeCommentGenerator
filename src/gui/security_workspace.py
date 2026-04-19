from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
)
from PyQt6.QtGui import QColor, QBrush, QFont
from PyQt6.QtCore import Qt
from src.gui.security_graph_widget import SecurityGraphWidget


class SecurityWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._security_report = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        title = QLabel("Security Analysis")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.safety_score_label = QLabel("Safety Score: N/A")
        self.safety_score_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #22c55e; padding: 8px 16px;"
            "background: rgba(34, 197, 94, 0.15); border-radius: 8px;"
        )
        header_layout.addWidget(self.safety_score_label)

        self.unsafe_pct_label = QLabel("0% unsafe")
        self.unsafe_pct_label.setStyleSheet(
            "font-size: 14px; color: #ef4444; padding: 6px 12px;"
            "background: rgba(239, 68, 68, 0.1); border-radius: 6px;"
        )
        header_layout.addWidget(self.unsafe_pct_label)

        layout.addLayout(header_layout)

        self.security_graph = SecurityGraphWidget()
        layout.addWidget(self.security_graph, stretch=3)

        issues_label = QLabel("Security Issues Detail")
        issues_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e2e8f0;")
        layout.addWidget(issues_label)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Severity", "Pattern", "Function", "Line", "Message / Remediation"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setStyleSheet("""
            QTableWidget { background: #1e293b; color: #f8fafc; gridline-color: #334155; border: none; font-size: 13px; }
            QHeaderView::section { background: #0f172a; color: #94a3b8; padding: 4px; border: 1px solid #334155; }
            QTableWidget::item:selected { background: #3b82f6; }
        """)
        layout.addWidget(self.table, stretch=2)

    def populate(self, security_report):
        self._security_report = security_report
        if not security_report:
            self.safety_score_label.setText("Safety Score: N/A")
            self.unsafe_pct_label.setText("N/A")
            self.security_graph.set_data(None)
            return

        safe_pct = security_report.module_safe_pct
        unsafe_pct = round(100.0 - safe_pct, 1)

        if safe_pct >= 80:
            color = "#22c55e"
            bg = "rgba(34, 197, 94, 0.15)"
        elif safe_pct >= 50:
            color = "#eab308"
            bg = "rgba(234, 179, 8, 0.15)"
        else:
            color = "#ef4444"
            bg = "rgba(239, 68, 68, 0.15)"

        self.safety_score_label.setText(f"Safety: {safe_pct}% safe")
        self.safety_score_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {color}; padding: 8px 16px;"
            f"background: {bg}; border-radius: 8px;"
        )
        self.unsafe_pct_label.setText(f"{unsafe_pct}% unsafe")
        self.unsafe_pct_label.setStyleSheet(
            f"font-size: 14px; color: {'#ef4444' if unsafe_pct > 20 else '#eab308' if unsafe_pct > 0 else '#22c55e'}; "
            f"padding: 6px 12px; border-radius: 6px;"
        )

        self.security_graph.set_data(security_report)

        self.table.setRowCount(0)
        if security_report.issues:
            self.table.setRowCount(len(security_report.issues))
            for i, issue in enumerate(security_report.issues):
                sev_item = QTableWidgetItem(issue.severity.upper())
                sev_colors = {"critical": "#ef4444", "high": "#f97316", "medium": "#eab308", "low": "#3b82f6"}
                sev_item.setForeground(QBrush(QColor(sev_colors.get(issue.severity, "#64748b"))))
                self.table.setItem(i, 0, sev_item)
                self.table.setItem(i, 1, QTableWidgetItem(issue.pattern_id))
                self.table.setItem(i, 2, QTableWidgetItem(issue.function_name))
                self.table.setItem(i, 3, QTableWidgetItem(str(issue.lineno)))
                msg = issue.message
                if issue.remediation:
                    msg += f" | Fix: {issue.remediation}"
                self.table.setItem(i, 4, QTableWidgetItem(msg))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
