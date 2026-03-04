"""
Data-Flow Analysis Engine — Week 8
====================================
Implements two classical data-flow analyses over a CFG:

1. Reaching Definitions (forward analysis)
   ─────────────────────────────────────────
   A definition d reaches a point p if there exists a path from d to p along
   which d is not re-defined (killed).

   GEN[B]  = definitions created in block B
   KILL[B] = definitions whose variable is re-defined in B
   IN[B]   = ∪{ OUT[P] : P ∈ predecessors(B) }
   OUT[B]  = GEN[B] ∪ (IN[B] − KILL[B])

2. Live Variable Analysis (backward analysis)
   ────────────────────────────────────────────
   A variable v is live at a point p if there exists a path from p to a use
   of v along which v is not re-defined.

   USE[B]  = variables used in B before any definition (upward-exposed uses)
   DEF[B]  = variables defined in B
   OUT[B]  = ∪{ IN[S] : S ∈ successors(B) }
   IN[B]   = USE[B] ∪ (OUT[B] − DEF[B])

After running both analyses the engine also computes:
  - unused_vars       : variables defined but never used across the whole function
  - used_before_assigned : variables loaded before any assignment in entry block
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from .cfg_builder import CFG, CFGNode
from ..ir.ir_nodes import IRInstruction, IROpcode, IRFunction


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DFAResult:
    """
    Complete data-flow analysis results for one function.

    Attributes:
        function_name        : Name of the analysed function.
        reaching_defs        : Map { block_label → set of definition names
                               reaching the *entry* of that block }.
        live_vars            : Map { block_label → set of variable names
                               live at the *entry* of that block }.
        used_before_assigned : Variables that appear as LOAD operands before
                               any ASSIGN / LOAD result in the entry block.
        unused_vars          : Variables defined (ASSIGN / LOAD result) anywhere
                               in the function but never appear as a LOAD operand.
    """
    function_name: str
    reaching_defs: Dict[str, Set[str]] = field(default_factory=dict)
    live_vars: Dict[str, Set[str]] = field(default_factory=dict)
    used_before_assigned: List[str] = field(default_factory=list)
    unused_vars: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Local gen / kill / use / def helpers
# ---------------------------------------------------------------------------

def _compute_gen_kill(node: CFGNode):
    """
    Compute GEN and KILL sets for reaching-definitions of a single block.

    A 'definition' is represented as the result temporary name (e.g. ``t3``).
    GEN[B] = { result : instr has a result }
    KILL[B] = set of all results defined later in another block that are
              re-defined here — we approximate this as the same GEN set
              (each re-definition kills prior definitions of the same name).
    """
    gen: Set[str] = set()
    for instr in node.instructions:
        if instr.result:
            gen.add(instr.result)
    # KILL ≈ same as GEN here (intra-block re-kills are absorbed)
    return gen, set(gen)


def _compute_use_def(node: CFGNode):
    """
    Compute USE and DEF sets for live-variable analysis of a single block.

    USE[B] = variables used (as operands of non-LABEL, non-JUMP instructions)
             before they are defined within the same block.
    DEF[B] = variables defined (result temporaries) in this block.
    """
    use: Set[str] = set()
    definition: Set[str] = set()
    for instr in node.instructions:
        if instr.op in (IROpcode.LABEL, IROpcode.JUMP, IROpcode.NOP):
            continue
        # Record uses before definition
        for operand in instr.operands:
            if operand and operand not in definition:
                use.add(operand)
        # Record definitions
        if instr.result:
            definition.add(instr.result)
    return use, definition


# ---------------------------------------------------------------------------
# Reaching Definitions (forward, worklist)
# ---------------------------------------------------------------------------

def _reaching_definitions(cfg: CFG) -> Dict[str, Set[str]]:
    """
    Compute reaching-definition IN-sets for all blocks.

    Returns:
        { label → set of definition names reaching the entry of that block }
    """
    gen_map: Dict[str, Set[str]] = {}
    kill_map: Dict[str, Set[str]] = {}
    out_map: Dict[str, Set[str]] = {}
    in_map:  Dict[str, Set[str]] = {}

    for label, node in cfg.nodes.items():
        gen, kill = _compute_gen_kill(node)
        gen_map[label]  = gen
        kill_map[label] = kill
        out_map[label]  = set(gen)  # initial OUT = GEN
        in_map[label]   = set()

    # Worklist: process all blocks; re-add if OUT changes
    worklist = list(cfg.nodes.keys())
    iterations = 0
    max_iter = len(cfg.nodes) * 10  # safety cap

    while worklist and iterations < max_iter:
        iterations += 1
        label = worklist.pop(0)
        node  = cfg.nodes[label]

        # IN[B] = union of OUT[P] for each predecessor P
        new_in: Set[str] = set()
        for pred_label in node.predecessors:
            new_in |= out_map.get(pred_label, set())

        # OUT[B] = GEN[B] ∪ (IN[B] − KILL[B])
        new_out = gen_map[label] | (new_in - kill_map[label])

        if new_out != out_map[label]:
            out_map[label] = new_out
            # Re-add successors
            for succ in node.successors:
                if succ not in worklist:
                    worklist.append(succ)

        in_map[label] = new_in

    return in_map


# ---------------------------------------------------------------------------
# Live Variable Analysis (backward, worklist)
# ---------------------------------------------------------------------------

def _live_variables(cfg: CFG) -> Dict[str, Set[str]]:
    """
    Compute live-variable IN-sets for all blocks.

    Returns:
        { label → set of variable names live at the entry of that block }
    """
    use_map: Dict[str, Set[str]] = {}
    def_map: Dict[str, Set[str]] = {}
    in_map:  Dict[str, Set[str]] = {}
    out_map: Dict[str, Set[str]] = {}

    for label, node in cfg.nodes.items():
        use, defn = _compute_use_def(node)
        use_map[label] = use
        def_map[label] = defn
        in_map[label]  = set(use)   # initial IN = USE
        out_map[label] = set()

    worklist = list(reversed(cfg.topological_order()))
    iterations = 0
    max_iter = len(cfg.nodes) * 10

    while worklist and iterations < max_iter:
        iterations += 1
        label = worklist.pop(0)
        node  = cfg.nodes[label]

        # OUT[B] = union of IN[S] for each successor S
        new_out: Set[str] = set()
        for succ_label in node.successors:
            new_out |= in_map.get(succ_label, set())

        # IN[B] = USE[B] ∪ (OUT[B] − DEF[B])
        new_in = use_map[label] | (new_out - def_map[label])

        if new_in != in_map[label]:
            in_map[label] = new_in
            for pred in node.predecessors:
                if pred not in worklist:
                    worklist.append(pred)

        out_map[label] = new_out

    return in_map


# ---------------------------------------------------------------------------
# Unused / used-before-assigned heuristics
# ---------------------------------------------------------------------------

def _find_unused_and_uninitialized(
    cfg: CFG,
    ir_function: IRFunction,
) -> tuple[List[str], List[str]]:
    """
    Scan all instructions across the whole function to find:
    - ``used_before_assigned``: names used as operands before appearing as
      results (checked only for LOAD instructions in the entry block, i.e.
      params that shadow potential globals).
    - ``unused_vars``: result temporaries that never appear as an operand
      anywhere in the function.
    """
    all_defs: Set[str]  = set()   # all result temp names
    all_uses: Set[str]  = set()   # all operand names

    for instr in ir_function.all_instructions():
        if instr.result:
            all_defs.add(instr.result)
        for op in instr.operands:
            if op and not op.startswith("__"):   # skip synthetic placeholders
                all_uses.add(op)

    # Unused: defined but never used as an operand
    # (exclude the final return temp — it's by definition only assigned once)
    unused = sorted(all_defs - all_uses - {f"t{i}" for i in range(len(ir_function.params))})

    # Used-before-assigned: operands that look like user variables (not temps)
    # that appear before any ASSIGN result in the entry block
    used_before: List[str] = []
    if ir_function.blocks:
        assigned_so_far: Set[str] = set()
        for instr in ir_function.blocks[0].instructions:
            if instr.op == IROpcode.ASSIGN and instr.result:
                assigned_so_far.add(instr.result)
            elif instr.op in (IROpcode.BINOP, IROpcode.CALL, IROpcode.RETURN):
                for op in instr.operands:
                    if (op and not op.startswith("t") and not op.startswith("__")
                            and op not in assigned_so_far):
                        used_before.append(op)

    return list(set(used_before)), unused


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dfa(cfg: CFG, ir_function: IRFunction) -> DFAResult:
    """
    Run reaching-definitions and live-variable analyses over a CFG.

    Args:
        cfg         : CFG built by ``cfg_builder.build_cfg()``.
        ir_function : The corresponding IRFunction (used for whole-function
                      unused-variable scan).

    Returns:
        DFAResult with all analysis data populated.
    """
    reaching = _reaching_definitions(cfg)
    live     = _live_variables(cfg)
    used_before, unused = _find_unused_and_uninitialized(cfg, ir_function)

    return DFAResult(
        function_name=ir_function.name,
        reaching_defs={label: set(s) for label, s in reaching.items()},
        live_vars={label: set(s) for label, s in live.items()},
        used_before_assigned=used_before,
        unused_vars=unused,
    )
