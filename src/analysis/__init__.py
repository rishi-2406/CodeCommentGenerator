"""
Analysis sub-package — Week 8
===============================
Control-flow analysis, data-flow analysis, and pattern detection.
"""
from .cfg_builder import build_cfg, CFGNode, CFG
from .dfa_engine import run_dfa, DFAResult
from .pattern_detector import detect_patterns, AnalysisReport, PatternFinding

__all__ = [
    "build_cfg", "CFGNode", "CFG",
    "run_dfa", "DFAResult",
    "detect_patterns", "AnalysisReport", "PatternFinding",
]
