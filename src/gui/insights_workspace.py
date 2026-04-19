from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QTextEdit, QLabel,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtGui import QColor, QBrush, QFont
from PyQt6.QtCore import Qt
from src.gui.syntax_highlighter import PythonSyntaxHighlighter
from src.gui.ast_graph_widget import AstGraphWidget
from src.gui.context_graph_widget import ContextGraphWidget
from src.gui.eval_graph_widget import EvalGraphWidget
from src.gui.security_graph_widget import SecurityGraphWidget


class InsightsWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Insights & Code Visualizer")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 12px; color: #e2e8f0;")
        layout.addWidget(title)

        self.tabs = QTabWidget()

        # 1. AST Features (Tree)
        self.tree_ast = QTreeWidget()
        self.tree_ast.setHeaderLabels(["AST Node", "Details"])
        self.tree_ast.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_ast.setStyleSheet("QTreeWidget { background: #1e293b; color: #f8fafc; border: none; font-size: 13px; } QHeaderView::section { background: #0f172a; color: #94a3b8; padding: 4px; border: 1px solid #334155; }")
        self.tabs.addTab(self.tree_ast, "AST Features")

        # 2. AST Graph
        self.ast_graph = AstGraphWidget()
        self.ast_graph.setStyleSheet("background-color: #0f172a;")
        self.tabs.addTab(self.ast_graph, "AST Graph")

        # 3. Context Graph (Tree)
        self.tree_ctx = QTreeWidget()
        self.tree_ctx.setHeaderLabels(["Function", "Context Data"])
        self.tree_ctx.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_ctx.setStyleSheet("QTreeWidget { background: #1e293b; color: #f8fafc; border: none; font-size: 13px; } QHeaderView::section { background: #0f172a; color: #94a3b8; padding: 4px; border: 1px solid #334155; }")
        self.tabs.addTab(self.tree_ctx, "Context Graph")

        # 4. Call Graph (Visual)
        self.call_graph = ContextGraphWidget()
        self.call_graph.setStyleSheet("background-color: #0f172a;")
        self.tabs.addTab(self.call_graph, "Call Graph")

        # 5. IR Dump (Text Editor with Syntax Highlighting)
        self.tab_ir = QTextEdit()
        self.tab_ir.setProperty("class", "CodeEditor")
        self.tab_ir.setReadOnly(True)
        self.tab_ir.setStyleSheet("QTextEdit { background: #1e293b; color: #f8fafc; font-family: monospace; font-size: 13px; border: none; }")
        self.hl_ir = PythonSyntaxHighlighter(self.tab_ir.document())
        self.tabs.addTab(self.tab_ir, "IR Dump")

        # 6. Analysis Report (Table)
        self.table_analysis = QTableWidget()
        self.table_analysis.setColumnCount(4)
        self.table_analysis.setHorizontalHeaderLabels(["Severity", "Pattern", "Function", "Message"])
        self.table_analysis.horizontalHeader().setStretchLastSection(True)
        self.table_analysis.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_analysis.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_analysis.setStyleSheet("""
            QTableWidget { background: #1e293b; color: #f8fafc; gridline-color: #334155; border: none; font-size: 13px; }
            QHeaderView::section { background: #0f172a; color: #94a3b8; padding: 4px; border: 1px solid #334155; }
            QTableWidget::item:selected { background: #3b82f6; }
        """)
        self.tabs.addTab(self.table_analysis, "Analysis Report")

        # 7. Evaluation Charts
        self.eval_graph = EvalGraphWidget()
        self.tabs.addTab(self.eval_graph, "Evaluation")

        # 8. Security Charts
        self.security_graph = SecurityGraphWidget()
        self.tabs.addTab(self.security_graph, "Security")

        layout.addWidget(self.tabs)

        self.tabs.setStyleSheet("""
            QTabBar::tab {
                background: #1e293b;
                color: #94a3b8;
                padding: 8px 16px;
                border: 1px solid #0f172a;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #3b82f6;
                color: white;
            }
            QTabWidget::pane {
                border: 1px solid #0f172a;
                background: #1e293b;
                border-radius: 4px;
            }
        """)

    def populate_insights(self, pipeline_results):
        mf = pipeline_results.get("mf")
        cg = pipeline_results.get("cg")
        ir_module = pipeline_results.get("ir")
        analysis_report = pipeline_results.get("analysis")
        security_report = pipeline_results.get("security_report")

        # Populate AST
        self.tree_ast.clear()
        if mf:
            root_funcs = QTreeWidgetItem(self.tree_ast, ["Functions", f"{len(mf.functions)} found"])
            for func in mf.functions:
                f_item = QTreeWidgetItem(root_funcs, [func.name, f"Line {func.lineno}"])

                if func.params:
                    p_root = QTreeWidgetItem(f_item, ["Parameters", ""])
                    for p in func.params:
                        p_desc = p.name
                        if p.annotation: p_desc += f": {p.annotation}"
                        if p.default: p_desc += f" = {p.default}"
                        QTreeWidgetItem(p_root, ["Param", p_desc])

                QTreeWidgetItem(f_item, ["Returns", func.return_annotation or "None"])
                QTreeWidgetItem(f_item, ["Complexity", f"Loops: {func.loops}, Conditionals: {func.conditionals}"])

            root_classes = QTreeWidgetItem(self.tree_ast, ["Classes", f"{len(mf.classes)} found"])
            for cls in mf.classes:
                c_item = QTreeWidgetItem(root_classes, [cls.name, f"Line {cls.lineno}"])
                if cls.bases: QTreeWidgetItem(c_item, ["Bases", ", ".join(cls.bases)])
                if cls.methods: QTreeWidgetItem(c_item, ["Methods", ", ".join(cls.methods)])

            self.tree_ast.expandAll()

            self.ast_graph.set_data(mf)

        # Populate Context
        self.tree_ctx.clear()
        if cg:
            for fc in cg.function_contexts:
                item = QTreeWidgetItem(self.tree_ctx, [fc.name, fc.complexity_label])

                if fc.complexity_label == "simple":
                    item.setForeground(1, QBrush(QColor("#22c55e")))
                elif fc.complexity_label == "moderate":
                    item.setForeground(1, QBrush(QColor("#eab308")))
                elif fc.complexity_label in ("complex", "very_complex"):
                    item.setForeground(1, QBrush(QColor("#ef4444")))

                if fc.calls_internal:
                    QTreeWidgetItem(item, ["Internal Calls", ", ".join(fc.calls_internal)])
                if fc.calls_external:
                    QTreeWidgetItem(item, ["External Calls", ", ".join(fc.calls_external)])

                security_issues = getattr(fc, 'security_issues', [])
                if security_issues:
                    sec_item = QTreeWidgetItem(item, ["Security Warnings", f"{len(security_issues)} found"])
                    sec_item.setForeground(0, QBrush(QColor("#ef4444")))
                    for sec in security_issues:
                        s_i = QTreeWidgetItem(sec_item, ["Warning", sec])
                        s_i.setForeground(1, QBrush(QColor("#ef4444")))
            self.tree_ctx.expandAll()

            self.call_graph.set_data(cg)

        # Populate IR Dump
        if ir_module:
            from src.ir import pretty_print_ir
            self.tab_ir.setPlainText(pretty_print_ir(ir_module))

        # Populate Analysis Report
        self.table_analysis.setRowCount(0)
        if analysis_report and analysis_report.findings:
            self.table_analysis.setRowCount(len(analysis_report.findings))
            for i, finding in enumerate(analysis_report.findings):
                sev_item = QTableWidgetItem(finding.severity.upper())
                if finding.severity == "warning":
                    sev_item.setForeground(QBrush(QColor("#eab308")))
                elif finding.severity == "error":
                    sev_item.setForeground(QBrush(QColor("#ef4444")))
                else:
                    sev_item.setForeground(QBrush(QColor("#3b82f6")))

                self.table_analysis.setItem(i, 0, sev_item)
                self.table_analysis.setItem(i, 1, QTableWidgetItem(finding.pattern_id))
                self.table_analysis.setItem(i, 2, QTableWidgetItem(finding.function_name))
                self.table_analysis.setItem(i, 3, QTableWidgetItem(finding.message))
        self.table_analysis.resizeColumnsToContents()
        self.table_analysis.horizontalHeader().setStretchLastSection(True)

        # Populate Evaluation Charts
        import json, os
        eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "eval_report.json")
        train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "training_report.json")
        self.eval_graph.set_data(eval_json_path=eval_path, training_json_path=train_path)

        # Populate Security Charts
        self.security_graph.set_data(security_report=security_report)
