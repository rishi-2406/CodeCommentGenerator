"""
Feature Vectors — Week 9 ML
============================
Converts a (FunctionFeature, FunctionContext) pair into a fixed-length
numeric numpy vector suitable for sklearn models.

Vector layout (26 dimensions):
  [0]  n_params             – number of non-self/cls parameters
  [1]  body_lines           – lines of code in function body
  [2]  loops                – for/while loop count
  [3]  conditionals         – if/elif count
  [4]  cyclomatic_complexity
  [5]  n_calls              – number of distinct function calls made
  [6]  is_method            – 1 if inside a class
  [7]  is_async             – 1 if async def
  [8]  has_return_annotation– 1 if return type annotated
  [9]  has_params_annotation– fraction of params that have annotations
  [10] complexity_simple    – one-hot: complexity == "simple"
  [11] complexity_moderate  – one-hot: complexity == "moderate"
  [12] complexity_complex   – one-hot: complexity == "complex"
  [13] complexity_very      – one-hot: complexity == "very_complex"
  [14..25] verb_bucket      – 12-D one-hot for first verb in name
"""
import re
import numpy as np
from typing import Optional

# Import from sibling modules (available in week9/src)
try:
    from ..ast_extractor import FunctionFeature, ParamFeature
    from ..context_analyzer import FunctionContext
except ImportError:
    # fallback when used standalone
    FunctionFeature = object  # type: ignore
    FunctionContext = object  # type: ignore


# ── Vocabulary of verb buckets ──────────────────────────────────────────────

_VERB_BUCKETS = [
    # (bucket_index, set_of_tokens)
    ("get_fetch",    {"get", "fetch", "load", "read", "retrieve"}),
    ("set_update",   {"set", "update", "write", "save", "store", "put"}),
    ("add_insert",   {"add", "append", "insert", "push", "register"}),
    ("remove",       {"remove", "delete", "clear", "reset", "pop", "drop"}),
    ("compute",      {"calc", "calculate", "compute", "count", "sum", "measure"}),
    ("find_search",  {"find", "search", "lookup", "query", "match"}),
    ("check_valid",  {"check", "validate", "verify", "is", "has", "can"}),
    ("parse_format", {"parse", "format", "convert", "transform", "encode", "decode"}),
    ("process_run",  {"process", "handle", "run", "execute", "start", "stop"}),
    ("build_create", {"build", "create", "make", "generate", "render", "init", "initialize", "setup"}),
    ("send_receive", {"send", "receive", "connect", "open", "close", "log"}),
    ("test_assert",  {"test", "assert", "mock", "spy", "stub"}),
]

VERB_BUCKET_NAMES = [name for name, _ in _VERB_BUCKETS]
_VERB_BUCKET_INDEX = {}
for _idx, (_name, _tokens) in enumerate(_VERB_BUCKETS):
    for _tok in _tokens:
        _VERB_BUCKET_INDEX[_tok] = _idx

# Total number of verb bucket dimensions
N_VERB_BUCKETS = len(_VERB_BUCKETS)

# Full feature names (for interpretability)
FEATURE_NAMES = [
    "n_params",
    "body_lines",
    "loops",
    "conditionals",
    "cyclomatic_complexity",
    "n_calls",
    "is_method",
    "is_async",
    "has_return_annotation",
    "frac_params_annotated",
    "complexity_simple",
    "complexity_moderate",
    "complexity_complex",
    "complexity_very_complex",
] + [f"verb_{name}" for name in VERB_BUCKET_NAMES]

FEATURE_DIM = len(FEATURE_NAMES)  # 26


# ── Helpers ──────────────────────────────────────────────────────────────────

def _split_name(name: str):
    """Split snake_case / CamelCase identifier into lowercase tokens."""
    parts = name.split("_")
    tokens = []
    for part in parts:
        sub = re.sub(r'([A-Z][a-z]+)', r' \1', part)
        sub = re.sub(r'([A-Z]+)([A-Z][a-z])', r' \1 \2', sub)
        tokens.extend(sub.strip().lower().split())
    return [t for t in tokens if t]


def _verb_one_hot(name: str) -> np.ndarray:
    """Return a one-hot vector (N_VERB_BUCKETS,) for the first matching verb."""
    vec = np.zeros(N_VERB_BUCKETS, dtype=np.float32)
    for tok in _split_name(name):
        if tok in _VERB_BUCKET_INDEX:
            vec[_VERB_BUCKET_INDEX[tok]] = 1.0
            break
    return vec


def _complexity_one_hot(label: str) -> np.ndarray:
    """Return a one-hot (4,) for complexity label."""
    mapping = {"simple": 0, "moderate": 1, "complex": 2, "very_complex": 3}
    vec = np.zeros(4, dtype=np.float32)
    idx = mapping.get(label, 0)
    vec[idx] = 1.0
    return vec


# ── Public API ────────────────────────────────────────────────────────────────

def extract_feature_vector(ff, fc=None) -> np.ndarray:
    """
    Extract a fixed-length numeric feature vector from a FunctionFeature
    and optional FunctionContext.

    Args:
        ff: FunctionFeature dataclass instance
        fc: FunctionContext dataclass instance (optional; zeroed if None)

    Returns:
        np.ndarray of shape (FEATURE_DIM,) = (26,) dtype float32
    """
    # Real params (skip self/cls)
    real_params = [p for p in ff.params if p.name not in ("self", "cls")]
    n_params = float(len(real_params))

    # Fraction of params that have type annotations
    if real_params:
        annotated = sum(1 for p in real_params if p.annotation)
        frac_annotated = annotated / len(real_params)
    else:
        frac_annotated = 0.0

    body_lines = float(max(ff.body_lines, 0))
    loops = float(ff.loops)
    conditionals = float(ff.conditionals)
    n_calls = float(len(ff.calls_made))
    is_method = float(ff.is_method)
    is_async = float(ff.is_async)
    has_return = float(ff.return_annotation is not None and ff.return_annotation != "None")

    # Context-derived features
    if fc is not None:
        cc = float(fc.cyclomatic_complexity)
        complexity_oh = _complexity_one_hot(fc.complexity_label)
    else:
        cc = 1.0
        complexity_oh = _complexity_one_hot("simple")

    base = np.array([
        n_params,
        body_lines,
        loops,
        conditionals,
        cc,
        n_calls,
        is_method,
        is_async,
        has_return,
        frac_annotated,
    ], dtype=np.float32)

    verb_oh = _verb_one_hot(ff.name)

    return np.concatenate([base, complexity_oh, verb_oh])
