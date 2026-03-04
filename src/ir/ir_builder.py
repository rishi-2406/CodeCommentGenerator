"""
IR Builder — Week 8
====================
Converts the Week 7 ModuleFeatures + ContextGraph into an IRModule.

Lowering strategy
-----------------
For every FunctionFeature we produce one IRFunction.  Each function gets a
sequence of IRBlocks generated from the structural information we already
have (parameter loads, variable assignments, calls, loops, branches, return).

Because we are lifting from high-level AST features (not bytecode or a full
parse-tree walk), the IR is *descriptive* rather than fully executable — it
faithfully captures the control-flow skeleton and data dependencies so that
the analysis passes (CFG, DFA, pattern detection) can reason about the code.

Builder algorithm per function
-------------------------------
1. ``entry`` block  — LABEL + one LOAD per parameter
2. For each call recorded in FunctionFeature.calls_made → CALL instruction
3. If loops > 0   → ``loop_N`` block with BRANCH (loop condition) + JUMP back
4. If conditionals > 0 → ``branch_N`` block  with BRANCH (if condition)
5. Variable assignments from ContextGraph.VariableInfo → ASSIGN instructions
6. ``exit`` block  — RETURN instruction
7. Wire successor / predecessor edges between blocks.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..ast_extractor import ModuleFeatures, FunctionFeature
from ..context_analyzer import ContextGraph, FunctionContext

from .ir_nodes import (
    IROpcode, IRInstruction, IRBlock, IRFunction, IRModule,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _TempCounter:
    """Simple sequential temporary-name generator: t0, t1, t2 …"""
    def __init__(self):
        self._n = 0

    def next(self) -> str:
        name = f"t{self._n}"
        self._n += 1
        return name


class _BlockBuilder:
    """Convenience wrapper for building a single IR basic block."""

    def __init__(self, label: str):
        self._block = IRBlock(label=label)

    def emit(self, op: IROpcode, result: Optional[str] = None,
             operands: Optional[List[str]] = None,
             lineno: int = 0,
             **meta) -> IRInstruction:
        instr = IRInstruction(
            op=op,
            result=result,
            operands=operands or [],
            lineno=lineno,
            meta={k: str(v) for k, v in meta.items()},
        )
        self._block.instructions.append(instr)
        return instr

    def build(self) -> IRBlock:
        return self._block


# ---------------------------------------------------------------------------
# Function lowering
# ---------------------------------------------------------------------------

def _lower_function(
    ff: FunctionFeature,
    fc: Optional[FunctionContext],
    temps: _TempCounter,
) -> IRFunction:
    """Lower one FunctionFeature + its optional FunctionContext into IRFunction."""

    ir_func = IRFunction(
        name=ff.name,
        params=[p.name for p in ff.params],
        return_type=ff.return_annotation,
        is_method=ff.is_method,
        is_async=ff.is_async,
        source_lineno=ff.lineno,
    )

    # ── 1. Entry block ──────────────────────────────────────────────────
    entry = _BlockBuilder("entry")
    entry.emit(IROpcode.LABEL, lineno=ff.lineno, name="entry")

    # ff.params is a List[ParamFeature]; use .name to get the string
    for param in ff.params:
        t = temps.next()
        entry.emit(IROpcode.LOAD, result=t, operands=[param.name], lineno=ff.lineno)

    entry_block = entry.build()
    ir_func.blocks.append(entry_block)

    # ── 2. Variable assignment block ─────────────────────────────────────
    if fc and fc.variables:
        assign_bb = _BlockBuilder("vars")
        assign_bb.emit(IROpcode.LABEL, lineno=ff.lineno, name="vars")
        for vi in fc.variables:
            if vi.assigned_at:
                t = temps.next()
                assign_bb.emit(
                    IROpcode.ASSIGN,
                    result=t,
                    operands=[vi.name],
                    lineno=vi.assigned_at[0],
                    inferred_type=vi.inferred_type or "unknown",
                )
        vars_block = assign_bb.build()
        ir_func.blocks.append(vars_block)
        entry_block.successors.append("vars")
        vars_block.predecessors.append("entry")
        prev_label = "vars"
    else:
        prev_label = "entry"

    # ── 3. Call block ────────────────────────────────────────────────────
    if ff.calls_made:
        calls_bb = _BlockBuilder("calls")
        calls_bb.emit(IROpcode.LABEL, lineno=ff.lineno, name="calls")
        for callee in ff.calls_made:
            t = temps.next()
            calls_bb.emit(
                IROpcode.CALL,
                result=t,
                operands=[callee],
                lineno=ff.lineno,
                callee=callee,
            )
        calls_block = calls_bb.build()
        ir_func.blocks.append(calls_block)
        ir_func.get_block(prev_label).successors.append("calls")  # type: ignore[union-attr]
        calls_block.predecessors.append(prev_label)
        prev_label = "calls"

    # ── 4. Loop blocks ──────────────────────────────────────────────────
    loop_lineno = ff.lineno + 1  # approximate; we don't have exact loop start
    for i in range(ff.loops):
        loop_label  = f"loop_{i}_header"
        body_label  = f"loop_{i}_body"
        after_label = f"loop_{i}_after"

        # Loop header: branch on condition
        header_bb = _BlockBuilder(loop_label)
        header_bb.emit(IROpcode.LABEL, lineno=loop_lineno, name=loop_label)
        t_cond = temps.next()
        header_bb.emit(IROpcode.BINOP, result=t_cond, operands=["__iter__", "__end__"],
                       lineno=loop_lineno, operator="lt")
        header_bb.emit(IROpcode.BRANCH, operands=[t_cond, body_label, after_label],
                       lineno=loop_lineno)
        header_block = header_bb.build()

        # Loop body
        body_bb = _BlockBuilder(body_label)
        body_bb.emit(IROpcode.LABEL, lineno=loop_lineno, name=body_label)
        t_body = temps.next()
        body_bb.emit(IROpcode.NOP, result=t_body, lineno=loop_lineno, note="loop_body")
        body_bb.emit(IROpcode.JUMP, operands=[loop_label], lineno=loop_lineno)
        body_block = body_bb.build()

        # After-loop placeholder
        after_bb = _BlockBuilder(after_label)
        after_bb.emit(IROpcode.LABEL, lineno=loop_lineno, name=after_label)
        after_block = after_bb.build()

        # Wire edges
        ir_func.get_block(prev_label).successors.append(loop_label)  # type: ignore[union-attr]
        header_block.predecessors.append(prev_label)
        header_block.successors  += [body_label, after_label]
        body_block.predecessors.append(loop_label)
        body_block.successors.append(loop_label)       # back-edge
        header_block.predecessors.append(body_label)   # back-edge target
        after_block.predecessors.append(loop_label)

        ir_func.blocks += [header_block, body_block, after_block]
        prev_label = after_label
        loop_lineno += 3  # rough spacing

    # ── 5. Branch blocks ────────────────────────────────────────────────
    branch_lineno = ff.lineno + 2
    for j in range(ff.conditionals):
        true_label  = f"branch_{j}_true"
        false_label = f"branch_{j}_false"
        join_label  = f"branch_{j}_join"

        # Condition evaluation
        cond_bb = _BlockBuilder(f"branch_{j}_cond")
        cond_bb.emit(IROpcode.LABEL, lineno=branch_lineno, name=f"branch_{j}_cond")
        t_cond = temps.next()
        cond_bb.emit(IROpcode.BINOP, result=t_cond, operands=["__cond_lhs__", "__cond_rhs__"],
                     lineno=branch_lineno, operator="eq")
        cond_bb.emit(IROpcode.BRANCH, operands=[t_cond, true_label, false_label],
                     lineno=branch_lineno)
        cond_block = cond_bb.build()

        true_bb = _BlockBuilder(true_label)
        true_bb.emit(IROpcode.LABEL, lineno=branch_lineno, name=true_label)
        t_true = temps.next()
        true_bb.emit(IROpcode.NOP, result=t_true, lineno=branch_lineno, note="then_branch")
        true_bb.emit(IROpcode.JUMP, operands=[join_label], lineno=branch_lineno)
        true_block = true_bb.build()

        false_bb = _BlockBuilder(false_label)
        false_bb.emit(IROpcode.LABEL, lineno=branch_lineno, name=false_label)
        t_false = temps.next()
        false_bb.emit(IROpcode.NOP, result=t_false, lineno=branch_lineno, note="else_branch")
        false_bb.emit(IROpcode.JUMP, operands=[join_label], lineno=branch_lineno)
        false_block = false_bb.build()

        join_bb = _BlockBuilder(join_label)
        join_bb.emit(IROpcode.LABEL, lineno=branch_lineno, name=join_label)
        phi_t = temps.next()
        join_bb.emit(IROpcode.PHI, result=phi_t, operands=[t_true, t_false],
                     lineno=branch_lineno)
        join_block = join_bb.build()

        # Wire edges
        ir_func.get_block(prev_label).successors.append(f"branch_{j}_cond")  # type: ignore[union-attr]
        cond_block.predecessors.append(prev_label)
        cond_block.successors   += [true_label, false_label]
        true_block.predecessors.append(f"branch_{j}_cond")
        true_block.successors.append(join_label)
        false_block.predecessors.append(f"branch_{j}_cond")
        false_block.successors.append(join_label)
        join_block.predecessors += [true_label, false_label]

        ir_func.blocks += [cond_block, true_block, false_block, join_block]
        prev_label = join_label
        branch_lineno += 4

    # ── 6. Exit block ────────────────────────────────────────────────────
    exit_bb = _BlockBuilder("exit")
    exit_bb.emit(IROpcode.LABEL, lineno=ff.lineno, name="exit")
    t_ret = temps.next()
    exit_bb.emit(
        IROpcode.RETURN,
        operands=[t_ret],
        lineno=ff.lineno + ff.body_lines,
        return_type=ff.return_annotation or "None",
    )
    exit_block = exit_bb.build()

    ir_func.get_block(prev_label).successors.append("exit")  # type: ignore[union-attr]
    exit_block.predecessors.append(prev_label)
    ir_func.blocks.append(exit_block)

    return ir_func


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ir(
    module_features: ModuleFeatures,
    context_graph: ContextGraph,
) -> IRModule:
    """
    Lower Week 7 analysis results into an IRModule.

    Args:
        module_features : Output of ``ast_extractor.extract_features()``.
        context_graph   : Output of ``context_analyzer.analyze_context()``.

    Returns:
        IRModule containing one IRFunction per Python function in the source.
    """
    ir_module = IRModule(source_file=module_features.filepath)
    temps = _TempCounter()

    # Build a lookup from function name → FunctionContext
    fc_map: Dict[str, FunctionContext] = {
        fc.name: fc for fc in context_graph.function_contexts
    }

    for ff in module_features.functions:
        fc = fc_map.get(ff.name)
        ir_func = _lower_function(ff, fc, temps)
        ir_module.functions.append(ir_func)

    return ir_module
