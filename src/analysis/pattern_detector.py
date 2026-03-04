"""
Pattern Detector — Week 8
==========================
Detects code-quality patterns / code smells from IR and DFA results.

Patterns detected
-----------------
  P001  unused_variable      (warning)  — variable assigned but never read
  P002  high_complexity      (warning)  — cyclomatic complexity > 10 (very_complex)
  P003  dead_block           (info)     — basic block with no predecessors (unreachable)
  P004  no_return_value      (info)     — non-void function has no RETURN operand
  P005  deeply_nested_loops  (warning)  — function has ≥ 3 loop blocks in IR
  P006  excessive_calls      (info)     — function calls > 10 distinct callees
  P007  missing_param_load   (info)     — parameter listed but no LOAD in entry block
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..ir.ir_nodes import IRModule, IRFunction, IROpcode
from .dfa_engine import DFAResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PatternFinding:
    """
    A single finding produced by the pattern detector.

    Attributes:
        pattern_id    : Short code like ``P001``.
        severity      : ``"info"``, ``"warning"``, or ``"error"``.
        function_name : Name of the function in which the pattern was detected.
        message       : Human-readable description.
        lineno        : Approximate source line (0 = unknown).
    """
    pattern_id: str
    severity: str          # "info" | "warning" | "error"
    function_name: str
    message: str
    lineno: int = 0

    def __repr__(self) -> str:
        return f"[{self.severity.upper()}] {self.pattern_id} in {self.function_name!r}: {self.message}"


@dataclass
class AnalysisReport:
    """
    Complete analysis report for all functions in an IRModule.

    Attributes:
        source_file : Path of the analysed source file.
        findings    : All patternFindings ordered by (function_name, lineno).
        summary     : Aggregate counts keyed by severity and pattern_id.
    """
    source_file: str = ""
    findings: List[PatternFinding] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def add(self, finding: PatternFinding) -> None:
        self.findings.append(finding)
        # Update summary
        self.summary[finding.severity] = self.summary.get(finding.severity, 0) + 1
        self.summary[finding.pattern_id] = self.summary.get(finding.pattern_id, 0) + 1

    def findings_for(self, function_name: str) -> List[PatternFinding]:
        return [f for f in self.findings if f.function_name == function_name]

    def __repr__(self) -> str:
        return f"AnalysisReport({self.source_file!r}, {len(self.findings)} findings)"


# ---------------------------------------------------------------------------
# Individual pattern checkers
# ---------------------------------------------------------------------------

def _check_unused_variables(
    ir_func: IRFunction,
    dfa: Optional[DFAResult],
    report: AnalysisReport,
) -> None:
    """P001 — variables assigned but never read."""
    if dfa is None:
        return
    for var in dfa.unused_vars:
        report.add(PatternFinding(
            pattern_id="P001",
            severity="warning",
            function_name=ir_func.name,
            message=f"Temporary {var!r} is assigned but never used as an operand.",
            lineno=ir_func.source_lineno,
        ))


def _check_high_complexity(
    ir_func: IRFunction,
    report: AnalysisReport,
) -> None:
    """P002 — excessive number of branch/loop blocks implies high CC."""
    branch_blocks = sum(
        1 for b in ir_func.blocks
        if "_cond" in b.label or "branch_" in b.label or "loop_" in b.label
    )
    # Heuristic: >10 branch-related blocks correlates with very_complex CC
    if branch_blocks > 10:
        report.add(PatternFinding(
            pattern_id="P002",
            severity="warning",
            function_name=ir_func.name,
            message=(
                f"Function has {branch_blocks} branch/loop IR blocks — "
                "likely very high cyclomatic complexity (> 10)."
            ),
            lineno=ir_func.source_lineno,
        ))


def _check_dead_blocks(
    ir_func: IRFunction,
    report: AnalysisReport,
) -> None:
    """P003 — blocks with no predecessors (except the entry block)."""
    for block in ir_func.blocks:
        if block.label == "entry":
            continue
        if not block.predecessors:
            report.add(PatternFinding(
                pattern_id="P003",
                severity="info",
                function_name=ir_func.name,
                message=f"Block {block.label!r} has no predecessors — potentially unreachable (dead code).",
                lineno=ir_func.source_lineno,
            ))


def _check_no_return_value(
    ir_func: IRFunction,
    report: AnalysisReport,
) -> None:
    """P004 — non-void function's RETURN has no operand."""
    if ir_func.return_type in (None, "None", ""):
        return   # void function — skip
    for block in ir_func.blocks:
        for instr in block.instructions:
            if instr.op == IROpcode.RETURN and not instr.operands:
                report.add(PatternFinding(
                    pattern_id="P004",
                    severity="info",
                    function_name=ir_func.name,
                    message=(
                        f"Function declares return type {ir_func.return_type!r} "
                        "but a RETURN instruction carries no value."
                    ),
                    lineno=instr.lineno,
                ))


def _check_deeply_nested_loops(
    ir_func: IRFunction,
    report: AnalysisReport,
) -> None:
    """P005 — three or more distinct loop_N_header blocks."""
    loop_headers = [b for b in ir_func.blocks if b.label.endswith("_header")]
    if len(loop_headers) >= 3:
        report.add(PatternFinding(
            pattern_id="P005",
            severity="warning",
            function_name=ir_func.name,
            message=(
                f"Function contains {len(loop_headers)} nested loop header blocks — "
                "consider refactoring to reduce loop depth."
            ),
            lineno=ir_func.source_lineno,
        ))


def _check_excessive_calls(
    ir_func: IRFunction,
    report: AnalysisReport,
) -> None:
    """P006 — function calls more than 10 distinct callees."""
    callees = set()
    for instr in ir_func.all_instructions():
        if instr.op == IROpcode.CALL and instr.meta.get("callee"):
            callees.add(instr.meta["callee"])
    if len(callees) > 10:
        report.add(PatternFinding(
            pattern_id="P006",
            severity="info",
            function_name=ir_func.name,
            message=(
                f"Function makes calls to {len(callees)} distinct callees — "
                "high coupling; consider splitting responsibilities."
            ),
            lineno=ir_func.source_lineno,
        ))


def _check_missing_param_load(
    ir_func: IRFunction,
    report: AnalysisReport,
) -> None:
    """P007 — parameter declared but no LOAD instruction in entry block."""
    loaded: set[str] = set()
    if ir_func.blocks:
        for instr in ir_func.blocks[0].instructions:
            if instr.op == IROpcode.LOAD and instr.operands:
                loaded.add(instr.operands[0])
    for param in ir_func.params:
        if param not in loaded and param != "self":
            report.add(PatternFinding(
                pattern_id="P007",
                severity="info",
                function_name=ir_func.name,
                message=f"Parameter {param!r} has no corresponding LOAD in the entry block.",
                lineno=ir_func.source_lineno,
            ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_patterns(
    ir_module: IRModule,
    dfa_results: Optional[List[DFAResult]] = None,
) -> AnalysisReport:
    """
    Run all pattern-detection checks over the IRModule.

    Args:
        ir_module   : IRModule produced by ``ir_builder.build_ir()``.
        dfa_results : Optional list of DFAResult objects (one per function)
                      produced by ``dfa_engine.run_dfa()``.  When supplied,
                      data-flow-based checks (P001) are also run.

    Returns:
        AnalysisReport with all discovered findings.
    """
    report = AnalysisReport(source_file=ir_module.source_file)

    # Build a lookup: function name → DFAResult
    dfa_map: Dict[str, DFAResult] = {}
    if dfa_results:
        for dr in dfa_results:
            dfa_map[dr.function_name] = dr

    for ir_func in ir_module.functions:
        dfa = dfa_map.get(ir_func.name)
        _check_unused_variables(ir_func, dfa, report)
        _check_high_complexity(ir_func, report)
        _check_dead_blocks(ir_func, report)
        _check_no_return_value(ir_func, report)
        _check_deeply_nested_loops(ir_func, report)
        _check_excessive_calls(ir_func, report)
        _check_missing_param_load(ir_func, report)

    # Sort findings by (function_name, lineno) for deterministic output
    report.findings.sort(key=lambda f: (f.function_name, f.lineno, f.pattern_id))

    return report
