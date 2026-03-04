"""
Comment Generator — Week 9 ML/AI Integration
=============================================
Generates meaningful code comments from ModuleFeatures and ContextGraph
using three complementary strategies:

  1. Rule-Based Engine    — deterministic, fast, always produces output
  2. Template NLP Engine  — richer English prose using token analysis
  3. ML Engine            — TF-IDF + LogReg / Template Ranking (Week 9)
                            Activated via ml_generate_comments().

Output: List[CommentItem] — each with location, text, and kind.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ast_extractor import ModuleFeatures, FunctionFeature, ClassFeature, ParamFeature
from .context_analyzer import ContextGraph, FunctionContext


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class CommentItem:
    node_id: str
    node_type: str       # "function" | "class" | "inline"
    lineno: int          # line BEFORE which the comment should be inserted
    col_offset: int      # indentation of the target node
    text: str            # the full comment text (may be multi-line docstring)
    kind: str            # "docstring" | "inline" | "block"
    target_name: str = ""


# ---------------------------------------------------------------------------
# NLP token utilities
# ---------------------------------------------------------------------------

# Common English stop words to skip when forming descriptions
_STOP_WORDS = {
    "a", "an", "the", "to", "of", "in", "for", "is", "it",
    "on", "at", "by", "as", "or", "be", "do", "if", "so",
    "up", "no", "my", "we", "us"
}

# Verb map: first meaningful token → verb phrase
_VERB_MAP = {
    "get":       "Retrieves",
    "fetch":     "Fetches",
    "load":      "Loads",
    "read":      "Reads",
    "set":       "Sets",
    "update":    "Updates",
    "write":     "Writes",
    "save":      "Saves",
    "store":     "Stores",
    "put":       "Puts",
    "add":       "Adds",
    "append":    "Appends",
    "insert":    "Inserts",
    "push":      "Pushes",
    "remove":    "Removes",
    "delete":    "Deletes",
    "clear":     "Clears",
    "reset":     "Resets",
    "pop":       "Pops",
    "calc":      "Calculates",
    "calculate": "Calculates",
    "compute":   "Computes",
    "count":     "Counts",
    "sum":       "Sums",
    "find":      "Finds",
    "search":    "Searches",
    "check":     "Checks",
    "validate":  "Validates",
    "verify":    "Verifies",
    "is":        "Checks whether",
    "has":       "Checks if",
    "can":       "Determines whether",
    "parse":     "Parses",
    "format":    "Formats",
    "convert":   "Converts",
    "transform": "Transforms",
    "encode":    "Encodes",
    "decode":    "Decodes",
    "process":   "Processes",
    "handle":    "Handles",
    "run":       "Runs",
    "execute":   "Executes",
    "start":     "Starts",
    "stop":      "Stops",
    "init":      "Initializes",
    "initialize":"Initializes",
    "setup":     "Sets up",
    "build":     "Builds",
    "create":    "Creates",
    "make":      "Creates",
    "generate":  "Generates",
    "render":    "Renders",
    "display":   "Displays",
    "show":      "Shows",
    "print":     "Prints",
    "log":       "Logs",
    "send":      "Sends",
    "receive":   "Receives",
    "connect":   "Connects",
    "open":      "Opens",
    "close":     "Closes",
    "sort":      "Sorts",
    "filter":    "Filters",
    "map":       "Maps",
    "merge":     "Merges",
    "split":     "Splits",
    "join":      "Joins",
    "extract":   "Extracts",
    "load_data": "Loads data from",
    "test":      "Tests",
    "assert":    "Asserts",
}


def _split_identifier(name: str) -> List[str]:
    """Split snake_case and CamelCase identifiers into lowercase tokens."""
    # First split on underscores
    parts = name.split("_")
    tokens = []
    for part in parts:
        # Then split on CamelCase boundaries
        sub = re.sub(r'([A-Z][a-z]+)', r' \1', part)
        sub = re.sub(r'([A-Z]+)([A-Z][a-z])', r' \1 \2', sub)
        tokens.extend(sub.strip().lower().split())
    return [t for t in tokens if t]


def _meaningful_tokens(name: str) -> List[str]:
    """Return non-stop-word tokens from an identifier."""
    return [t for t in _split_identifier(name) if t not in _STOP_WORDS]


def _pick_verb(tokens: List[str]) -> str:
    """Pick the best verb phrase for a list of name tokens."""
    for t in tokens:
        if t in _VERB_MAP:
            return _VERB_MAP[t]
    return "Handles"


def _param_description(params: List[ParamFeature]) -> str:
    """Build a human-readable list of parameters."""
    if not params:
        return ""
    param_strs = []
    for p in params:
        if p.name in ("self", "cls"):
            continue
        desc = p.name
        if p.annotation:
            desc += f" ({p.annotation})"
        param_strs.append(desc)
    if not param_strs:
        return ""
    return ", ".join(param_strs)


# ---------------------------------------------------------------------------
# Rule-based engine
# ---------------------------------------------------------------------------

def _generate_function_docstring(ff: FunctionFeature, fc: Optional[FunctionContext] = None) -> str:
    """
    Generate a docstring for a function using rule-based + template NLP.
    """
    tokens = _meaningful_tokens(ff.name)
    verb = _pick_verb(tokens)
    noun_phrase = " ".join(t for t in tokens if t not in _VERB_MAP).strip()
    if not noun_phrase:
        noun_phrase = ff.name.replace("_", " ")

    # Summary line
    summary = f"{verb} {noun_phrase}."

    lines = ['"""', summary]

    # Parameters section
    real_params = [p for p in ff.params if p.name not in ("self", "cls")]
    if real_params:
        lines.append("")
        lines.append("Args:")
        for p in real_params:
            ann = f" ({p.annotation})" if p.annotation else ""
            default = f", defaults to {p.default}" if p.default is not None else ""
            lines.append(f"    {p.name}{ann}: {p.name.replace('_', ' ')}{default}.")

    # Returns section
    if ff.return_annotation and ff.return_annotation != "None":
        lines.append("")
        lines.append("Returns:")
        lines.append(f"    {ff.return_annotation}: The result of {noun_phrase}.")

    # Complexity note
    if fc and fc.complexity_label in ("complex", "very_complex"):
        lines.append("")
        lines.append(f"Note:")
        lines.append(f"    Cyclomatic complexity is {fc.cyclomatic_complexity} ({fc.complexity_label}).")

    lines.append('"""')
    return "\n".join(lines)


def _generate_class_docstring(cf: ClassFeature) -> str:
    """Generate a docstring for a class."""
    tokens = _meaningful_tokens(cf.name)
    noun_phrase = " ".join(tokens).strip() or cf.name
    # Capitalise first letter of noun_phrase
    noun_phrase = noun_phrase[0].upper() + noun_phrase[1:] if noun_phrase else cf.name

    lines = ['"""', f"Represents a {noun_phrase}."]

    if cf.bases and cf.bases != ["object"]:
        lines.append(f"Inherits from: {', '.join(cf.bases)}.")

    if cf.methods:
        lines.append("")
        public_methods = [m for m in cf.methods if not m.startswith("_")]
        if public_methods:
            lines.append("Methods:")
            for m in public_methods[:6]:  # cap at 6 methods listed
                m_verb = _pick_verb(_meaningful_tokens(m))
                m_noun = " ".join(t for t in _meaningful_tokens(m) if t not in _VERB_MAP)
                lines.append(f"    {m}(): {m_verb} {m_noun}.")

    lines.append('"""')
    return "\n".join(lines)


def _generate_inline_comment(ff: FunctionFeature, fc: FunctionContext) -> Optional[str]:
    """
    Generate a block comment above a complex function body
    (used when function has moderate or higher complexity and no docstring).
    """
    if fc.complexity_label == "simple":
        return None
    tokens = _meaningful_tokens(ff.name)
    verb = _pick_verb(tokens)
    noun = " ".join(t for t in tokens if t not in _VERB_MAP).strip() or ff.name
    cc = fc.cyclomatic_complexity
    return (
        f"# {verb} {noun}.\n"
        f"# Cyclomatic complexity: {cc} ({fc.complexity_label}). "
        f"Contains {ff.loops} loop(s) and {ff.conditionals} conditional(s)."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_comments(
    module_features: ModuleFeatures,
    context_graph: ContextGraph,
) -> List[CommentItem]:

    comments: List[CommentItem] = []

    # Build quick lookup: function name -> FunctionContext
    fc_map: Dict[str, FunctionContext] = {
        fc.name: fc for fc in context_graph.function_contexts
    }

    # --- Classes ---
    for cf in module_features.classes:
        if not cf.has_docstring:
            doc_text = _generate_class_docstring(cf)
            comments.append(CommentItem(
                node_id=cf.node_id,
                node_type="class",
                lineno=cf.lineno,
                col_offset=cf.col_offset,
                text=doc_text,
                kind="docstring",
                target_name=cf.name,
            ))

    # --- Functions ---
    for ff in module_features.functions:
        fc = fc_map.get(ff.name)

        if not ff.has_docstring:
            doc_text = _generate_function_docstring(ff, fc)
            comments.append(CommentItem(
                node_id=ff.node_id,
                node_type="function",
                lineno=ff.lineno,
                col_offset=ff.col_offset,
                text=doc_text,
                kind="docstring",
                target_name=ff.name,
            ))

        # Inline block comment for moderate/complex/very_complex functions
        if fc and fc.complexity_label != "simple":
            inline = _generate_inline_comment(ff, fc)
            if inline:
                comments.append(CommentItem(
                    node_id=ff.node_id + "_inline",
                    node_type="inline",
                    lineno=ff.lineno,
                    col_offset=ff.col_offset,
                    text=inline,
                    kind="inline",
                    target_name=ff.name,
                ))

    # Sort by lineno so attacher processes in order
    comments.sort(key=lambda c: (c.lineno, c.kind))
    return comments


# ---------------------------------------------------------------------------
# ML-enhanced API (Week 9)
# ---------------------------------------------------------------------------

def ml_generate_comments(
    module_features: ModuleFeatures,
    context_graph: ContextGraph,
    model_selector=None,
) -> List[CommentItem]:
    """
    Generate comments using the ML ModelSelector when available.

    For each function without a docstring:
      1. Extract its feature vector.
      2. First obtain the rule-based comment as a fallback.
      3. Ask the ModelSelector to predict; use its output if confidence ≥ threshold.
      4. Wrap the result in a CommentItem (same format as generate_comments).

    Falls back to the pure rule-based generate_comments() if no
    ModelSelector is provided or if a model error occurs.

    Args:
        module_features: Output of ast_extractor.extract_features().
        context_graph:   Output of context_analyzer.analyze_context().
        model_selector:  A ModelSelector instance (or None).

    Returns:
        List[CommentItem] sorted by line number.
    """
    if model_selector is None or not model_selector.is_ready():
        return generate_comments(module_features, context_graph)

    try:
        from .ml.feature_vectors import extract_feature_vector
    except ImportError:
        return generate_comments(module_features, context_graph)

    # First generate rule-based comments to get fallback text
    rule_comments = generate_comments(module_features, context_graph)
    rule_map: Dict[str, CommentItem] = {
        c.target_name: c for c in rule_comments if c.kind == "docstring"
    }

    fc_map: Dict[str, FunctionContext] = {
        fc.name: fc for fc in context_graph.function_contexts
    }

    comments: List[CommentItem] = []

    # ------ Classes (rule-based for now; ML focuses on functions) ----------
    for cf in module_features.classes:
        if not cf.has_docstring:
            doc_text = _generate_class_docstring(cf)
            comments.append(CommentItem(
                node_id=cf.node_id,
                node_type="class",
                lineno=cf.lineno,
                col_offset=cf.col_offset,
                text=doc_text,
                kind="docstring",
                target_name=cf.name,
            ))

    # ------ Functions — ML prediction ------------------------------------
    for ff in module_features.functions:
        fc = fc_map.get(ff.name)

        if not ff.has_docstring:
            fallback_item = rule_map.get(ff.name)
            fallback_text = fallback_item.text if fallback_item else None

            try:
                feat_vec = extract_feature_vector(ff, fc)
                ml_text, source, confidence = model_selector.predict(
                    ff.name, feat_vec, fallback=fallback_text
                )
                # Ensure it looks like a proper docstring
                if not ml_text.strip().startswith('"""'):
                    ml_text = f'"""{ml_text.strip()}"""'
                doc_text = ml_text
            except Exception:
                doc_text = fallback_text or _generate_function_docstring(ff, fc)

            comments.append(CommentItem(
                node_id=ff.node_id,
                node_type="function",
                lineno=ff.lineno,
                col_offset=ff.col_offset,
                text=doc_text,
                kind="docstring",
                target_name=ff.name,
            ))

        # Inline block comment for moderate/complex/very_complex functions
        if fc and fc.complexity_label != "simple":
            inline = _generate_inline_comment(ff, fc)
            if inline:
                comments.append(CommentItem(
                    node_id=ff.node_id + "_inline",
                    node_type="inline",
                    lineno=ff.lineno,
                    col_offset=ff.col_offset,
                    text=inline,
                    kind="inline",
                    target_name=ff.name,
                ))

    comments.sort(key=lambda c: (c.lineno, c.kind))
    return comments
