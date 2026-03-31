"""
Context Analyzer — Week 7 Core Engine
======================================
Enriches ModuleFeatures with semantic analysis:
- Cyclomatic complexity per function
- Variable lifespan tracking (assigned vs. used)
- Call graph (which functions call which within the module)
- Simple type inference heuristics

Output: ContextGraph dict that feeds into comment_generator.
"""
import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .ast_extractor import ModuleFeatures, FunctionFeature, ClassFeature


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VariableInfo:
    name: str
    assigned_at: List[int] = field(default_factory=list)  # line numbers
    used_at: List[int] = field(default_factory=list)
    inferred_type: Optional[str] = None


@dataclass
class FunctionContext:
    node_id: str
    name: str
    cyclomatic_complexity: int = 1
    variables: List[VariableInfo] = field(default_factory=list)
    calls_external: List[str] = field(default_factory=list)   # calls to module functions
    calls_internal: List[str] = field(default_factory=list)   # calls to other module functions
    complexity_label: str = "simple"   # simple / moderate / complex / very_complex
    security_issues: List[str] = field(default_factory=list)


@dataclass
class ContextGraph:
    module_name: str = ""
    function_contexts: List[FunctionContext] = field(default_factory=list)
    call_graph: Dict[str, List[str]] = field(default_factory=dict)   # caller -> [callee, ...]
    module_function_names: Set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Cyclomatic complexity visitor
# ---------------------------------------------------------------------------

class CyclomaticComplexityVisitor(ast.NodeVisitor):
    """
    Computes McCabe cyclomatic complexity for a single function body.
    Complexity = 1 + number of branching nodes (if, elif, for, while,
                                                 except, with, assert, comprehension).
    """
    BRANCH_NODES = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
    )

    def __init__(self):
        self.complexity = 1

    def visit_If(self, node: ast.If):
        self.complexity += 1
        # Each elif also adds 1 — handled by orelse being another If
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        self.complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        # Each and/or operand beyond the first adds a branch
        self.complexity += len(node.values) - 1
        self.generic_visit(node)


def compute_cyclomatic_complexity(func_node_body: List[ast.stmt]) -> int:
    """Compute cyclomatic complexity for a list of AST statements."""
    visitor = CyclomaticComplexityVisitor()
    # Wrap in a temporary module to allow generic_visit to work
    dummy = ast.Module(body=func_node_body, type_ignores=[])
    visitor.visit(dummy)
    return visitor.complexity


# ---------------------------------------------------------------------------
# Variable tracker
# ---------------------------------------------------------------------------

class VariableTracker(ast.NodeVisitor):
    """
    Tracks variable assignment and usage within a function body.
    """

    def __init__(self):
        self._vars: Dict[str, VariableInfo] = {}

    def _get_or_create(self, name: str) -> VariableInfo:
        if name not in self._vars:
            self._vars[name] = VariableInfo(name=name)
        return self._vars[name]

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                vi = self._get_or_create(target.id)
                vi.assigned_at.append(node.lineno)
                # Infer type from right-hand side
                if vi.inferred_type is None:
                    vi.inferred_type = _infer_type(node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            vi = self._get_or_create(node.target.id)
            vi.assigned_at.append(node.lineno)
            try:
                vi.inferred_type = ast.unparse(node.annotation)
            except Exception:
                pass
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            vi = self._get_or_create(node.id)
            vi.used_at.append(node.lineno)
        self.generic_visit(node)

    def get_variables(self) -> List[VariableInfo]:
        return list(self._vars.values())


def _infer_type(node: ast.expr) -> Optional[str]:
    """Simple type inference from literals."""
    if isinstance(node, ast.Constant):
        return type(node.value).__name__
    elif isinstance(node, ast.List):
        return "list"
    elif isinstance(node, ast.Dict):
        return "dict"
    elif isinstance(node, ast.Set):
        return "set"
    elif isinstance(node, ast.Tuple):
        return "tuple"
    elif isinstance(node, ast.Call):
        try:
            return ast.unparse(node.func)
        except Exception:
            return "object"
    return None


# ---------------------------------------------------------------------------
# Context analyzer
# ---------------------------------------------------------------------------

def _complexity_label(cc: int) -> str:
    if cc <= 2:
        return "simple"
    elif cc <= 5:
        return "moderate"
    elif cc <= 10:
        return "complex"
    else:
        return "very_complex"


def analyze_context(
    module_features: ModuleFeatures,
    ast_tree: ast.AST,
    source_code: str = ""
) -> ContextGraph:
    """
    Analyze a ModuleFeatures object and produce a ContextGraph.

    Args:
        module_features: Output of ast_extractor.extract_features()
        ast_tree: The original parsed AST (for walking function bodies)
        source_code: Raw source string (unused here, kept for API compatibility)

    Returns:
        ContextGraph with per-function contexts and call graph
    """
    # Build a map: function name -> AST FunctionDef node
    func_nodes: Dict[str, ast.FunctionDef] = {}
    for node in ast.walk(ast_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_nodes[node.name] = node

    module_func_names: Set[str] = {f.name for f in module_features.functions}
    cg = ContextGraph(
        module_name=module_features.filepath,
        module_function_names=module_func_names,
    )

    for ff in module_features.functions:
        # Retrieve AST node for this function
        ast_node = func_nodes.get(ff.name)
        if ast_node is None:
            continue

        # Cyclomatic complexity
        cc = compute_cyclomatic_complexity(ast_node.body)

        # Variable tracking
        tracker = VariableTracker()
        for stmt in ast_node.body:
            tracker.visit(stmt)
        variables = tracker.get_variables()

        # Classify calls as internal (to module functions) or external
        internal_calls = [c for c in ff.calls_made if c.split('.')[0] in module_func_names and c != ff.name]
        external_calls = [c for c in ff.calls_made if c.split('.')[0] not in module_func_names]

        # Basic Security Analysis
        security_issues = []
        for call in ff.calls_made:
            if call in ("eval", "exec", "compile"):
                security_issues.append(f"Uses dangerous builtin '{call}'")
            elif "subprocess" in call and "shell=True" in source_code[ff.lineno-1:ff.lineno+ff.body_lines]:
                # Heuristic: If subprocess is called and shell=True appears in body
                security_issues.append("Potential shell injection risk (subprocess with shell=True)")
            elif "md5" in call or "sha1" in call:
                security_issues.append(f"Uses weak cryptographic hash function '{call}'")

        # Basic heuristic for hardcoded passwords (very simple check on variable assignments)
        for v in variables:
            low_name = v.name.lower()
            if "password" in low_name or "secret" in low_name or "token" in low_name:
                if v.inferred_type == "str":
                    security_issues.append(f"Potential hardcoded secret assigned to '{v.name}'")

        fc = FunctionContext(
            node_id=ff.node_id,
            name=ff.name,
            cyclomatic_complexity=cc,
            variables=variables,
            calls_internal=internal_calls,
            calls_external=external_calls,
            complexity_label=_complexity_label(cc),
            security_issues=security_issues,
        )
        cg.function_contexts.append(fc)
        cg.call_graph[ff.name] = ff.calls_made

    return cg


def context_to_dict(cg: ContextGraph) -> Dict[str, Any]:
    """Serialize ContextGraph to a plain dict (JSON-compatible)."""
    import dataclasses

    def _convert(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert(i) for i in obj]
        elif isinstance(obj, set):
            return sorted(list(obj))
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        else:
            return obj

    return _convert(cg)
