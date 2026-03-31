from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QTextEdit, QLabel,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtGui import QColor, QBrush, QFont
from PyQt6.QtCore import Qt
from src.gui.syntax_highlighter import PythonSyntaxHighlighter

class InsightsWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        title = QLabel("📈 Insights & Code Visualizer")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 12px; color: #e2e8f0;")
        layout.addWidget(title)
        
        self.tabs = QTabWidget()
        
        # 1. AST Features (Tree)
        self.tree_ast = QTreeWidget()
        self.tree_ast.setHeaderLabels(["AST Node", "Details"])
        self.tree_ast.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_ast.setStyleSheet("QTreeWidget { background: #1e293b; color: #f8fafc; border: none; font-size: 13px; } QHeaderView::section { background: #0f172a; color: #94a3b8; padding: 4px; border: 1px solid #334155; }")
        self.tabs.addTab(self.tree_ast, "AST Features")
        
        # 2. Context Graph (Tree)
        self.tree_ctx = QTreeWidget()
        self.tree_ctx.setHeaderLabels(["Function", "Context Data"])
        self.tree_ctx.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_ctx.setStyleSheet("QTreeWidget { background: #1e293b; color: #f8fafc; border: none; font-size: 13px; } QHeaderView::section { background: #0f172a; color: #94a3b8; padding: 4px; border: 1px solid #334155; }")
        self.tabs.addTab(self.tree_ctx, "Context Graph")
        
        # 3. IR Dump (Text Editor with Syntax Highlighting)
        self.tab_ir = QTextEdit()
        self.tab_ir.setProperty("class", "CodeEditor")
        self.tab_ir.setReadOnly(True)
        self.tab_ir.setStyleSheet("QTextEdit { background: #1e293b; color: #f8fafc; font-family: monospace; font-size: 13px; border: none; }")
        self.hl_ir = PythonSyntaxHighlighter(self.tab_ir.document()) 
        self.tabs.addTab(self.tab_ir, "IR Dump")
        
        # 4. Analysis Report (Table)
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
        
        layout.addWidget(self.tabs)
        
        # Set stylesheet for dark tabs
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
        
        # Populate AST
        self.tree_ast.clear()
        if mf:
            root_funcs = QTreeWidgetItem(self.tree_ast, ["Functions", f"{len(mf.functions)} found"])
            for func in mf.functions:
                f_item = QTreeWidgetItem(root_funcs, [func.name, f"Line {func.lineno}"])
                
                # Parameters
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
            
        # Populate Context
        self.tree_ctx.clear()
        if cg:
            for fc in cg.function_contexts:
                item = QTreeWidgetItem(self.tree_ctx, [fc.name, fc.complexity_label])
                
                # Colors based on complexity
                if fc.complexity_label == "simple":
                    item.setForeground(1, QBrush(QColor("#22c55e"))) # Green
                elif fc.complexity_label == "moderate":
                    item.setForeground(1, QBrush(QColor("#eab308"))) # Yellow
                elif fc.complexity_label in ("complex", "very_complex"):
                    item.setForeground(1, QBrush(QColor("#ef4444"))) # Red
                    
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
                # Severity colors
                if finding.severity == "warning":
                    sev_item.setForeground(QBrush(QColor("#eab308")))
                elif finding.severity == "error":
                    sev_item.setForeground(QBrush(QColor("#ef4444")))
                else:
                    sev_item.setForeground(QBrush(QColor("#3b82f6"))) # Info
                    
                self.table_analysis.setItem(i, 0, sev_item)
                self.table_analysis.setItem(i, 1, QTableWidgetItem(finding.pattern_id))
                self.table_analysis.setItem(i, 2, QTableWidgetItem(finding.function_name))
                self.table_analysis.setItem(i, 3, QTableWidgetItem(finding.message))
        self.table_analysis.resizeColumnsToContents()
        self.table_analysis.horizontalHeader().setStretchLastSection(True)
