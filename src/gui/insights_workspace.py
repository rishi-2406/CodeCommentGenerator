from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QTextEdit, QLabel
from src.gui.syntax_highlighter import JsonSyntaxHighlighter, PythonSyntaxHighlighter

class InsightsWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        title = QLabel("📈 Insights & Code Visualizer")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 12px;")
        layout.addWidget(title)
        
        self.tabs = QTabWidget()
        
        self.tab_ast = QTextEdit()
        self.tab_ast.setProperty("class", "CodeEditor")
        self.tab_ast.setReadOnly(True)
        self.hl_ast = JsonSyntaxHighlighter(self.tab_ast.document())
        self.tabs.addTab(self.tab_ast, "AST Features")
        
        self.tab_ctx = QTextEdit()
        self.tab_ctx.setProperty("class", "CodeEditor")
        self.tab_ctx.setReadOnly(True)
        self.hl_ctx = JsonSyntaxHighlighter(self.tab_ctx.document())
        self.tabs.addTab(self.tab_ctx, "Context Graph")
        
        self.tab_ir = QTextEdit()
        self.tab_ir.setProperty("class", "CodeEditor")
        self.tab_ir.setReadOnly(True)
        self.hl_ir = PythonSyntaxHighlighter(self.tab_ir.document()) # IR looks somewhat like python
        self.tabs.addTab(self.tab_ir, "IR Dump")
        
        self.tab_analysis = QTextEdit()
        self.tab_analysis.setProperty("class", "CodeEditor")
        self.tab_analysis.setReadOnly(True)
        # Analysis report can remain plain text since it's a custom line-based format
        self.tabs.addTab(self.tab_analysis, "Analysis Report")
        
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
            }
            QTabBar::tab:selected {
                background: #197fe6;
                color: white;
            }
            QTabWidget::pane {
                border: 1px solid #0f172a;
                background: #1e293b;
            }
        """)
        
    def populate_insights(self, pipeline_results):
        from src.ast_extractor import features_to_dict
        from src.context_analyzer import context_to_dict
        from src.ir import pretty_print_ir
        import json
        
        mf = pipeline_results.get("mf")
        cg = pipeline_results.get("cg")
        ir_module = pipeline_results.get("ir")
        analysis_report = pipeline_results.get("analysis")
        
        if mf: self.tab_ast.setPlainText(json.dumps(features_to_dict(mf), indent=2))
        if cg: self.tab_ctx.setPlainText(json.dumps(context_to_dict(cg), indent=2))
        if ir_module: self.tab_ir.setPlainText(pretty_print_ir(ir_module))
        
        if analysis_report:
            report_lines = []
            if analysis_report.findings:
                for finding in analysis_report.findings:
                    report_lines.append(f"[{finding.severity.upper()}] {finding.pattern_id} @ {finding.function_name}: {finding.message}")
            else:
                report_lines.append("No patterns detected.")
            report_lines.append(f"\nSummary: {analysis_report.summary}")
            self.tab_analysis.setPlainText("\n".join(report_lines))
