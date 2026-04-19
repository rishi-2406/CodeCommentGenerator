import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..ast_extractor import FunctionFeature, ClassFeature
from ..context_analyzer import FunctionContext
from ..ast_body_extractor import extract_raises, extract_returned_types
from ..comment_generator import (
    _meaningful_tokens,
    _pick_verb,
    _humanize,
    _sanitize_docstring_content,
    _describe_body,
    _describe_class_attributes,
    CommentItem,
    _STOP_WORDS,
)


@dataclass
class SymbolicRule:
    pattern_id: str
    condition: str
    description: str
    severity: str = "info"


class SymbolicReasoner:
    """
    Symbolic knowledge base that provides pattern-to-description rules
    and constraint validation for neurosymbolic comment generation.
    """

    def __init__(self):
        self._rules = self._build_rules()

    def _build_rules(self) -> List[SymbolicRule]:
        return [
            SymbolicRule("SYM001", "loop_and_sort", "Sorts and iterates over items using a loop.", "info"),
            SymbolicRule("SYM002", "property_accessor", "Property accessor for an attribute.", "info"),
            SymbolicRule("SYM003", "static_method", "Static method with no instance binding.", "info"),
            SymbolicRule("SYM004", "class_method", "Class method that receives the class as first argument.", "info"),
            SymbolicRule("SYM005", "async_network", "Asynchronously fetches data from a remote source.", "info"),
            SymbolicRule("SYM006", "dangerous_eval", "DANGER: Executes dynamic code — severe security risk.", "warning"),
            SymbolicRule("SYM007", "dangerous_exec", "DANGER: Executes dynamic code — severe security risk.", "warning"),
            SymbolicRule("SYM008", "dangerous_compile", "DANGER: Compiles and executes dynamic code.", "warning"),
            SymbolicRule("SYM009", "hardcoded_secret", "DANGER: Potential hardcoded secret in source.", "warning"),
            SymbolicRule("SYM010", "iterator_pattern", "Yields items one at a time (generator/iterator pattern).", "info"),
            SymbolicRule("SYM011", "context_manager", "Context manager supporting the 'with' statement.", "info"),
            SymbolicRule("SYM012", "recursive_pattern", "Recursive function that calls itself.", "info"),
            SymbolicRule("SYM013", "high_complexity", "High control-flow complexity — consider refactoring.", "warning"),
        ]

    def match_rules(self, ff: FunctionFeature, fc: Optional[FunctionContext]) -> List[SymbolicRule]:
        matched = []
        network_calls = {"requests.get", "requests.post", "httpx.get", "httpx.post",
                         "urlopen", "urllib.request.urlopen", "aiohttp"}
        sort_calls = {"sorted", "list.sort", "sort"}

        calls_set = set(ff.calls_made)

        has_loop_and_sort = ff.loops > 0 and bool(calls_set & sort_calls)
        if has_loop_and_sort:
            matched.append(self._find_rule("SYM001"))

        for dec in ff.decorators:
            if "property" in dec:
                matched.append(self._find_rule("SYM002"))
            elif "staticmethod" in dec:
                matched.append(self._find_rule("SYM003"))
            elif "classmethod" in dec:
                matched.append(self._find_rule("SYM004"))

        if ff.is_async:
            ext_calls = fc.calls_external if fc else []
            for c in ext_calls:
                c_lower = c.lower()
                if any(n in c_lower for n in network_calls):
                    matched.append(self._find_rule("SYM005"))
                    break

        for call in calls_set:
            if call == "eval":
                matched.append(self._find_rule("SYM006"))
            elif call == "exec":
                matched.append(self._find_rule("SYM007"))
            elif call == "compile":
                matched.append(self._find_rule("SYM008"))

        if fc and fc.security_issues:
            for issue in fc.security_issues:
                if "secret" in issue.lower() or "password" in issue.lower() or "token" in issue.lower():
                    matched.append(self._find_rule("SYM009"))
                    break

        for dec in ff.decorators:
            if "contextmanager" in dec.lower() or "asynccontextmanager" in dec.lower():
                matched.append(self._find_rule("SYM011"))

        if ff.name and calls_set and ff.name in calls_set:
            matched.append(self._find_rule("SYM012"))

        if fc and fc.complexity_label in ("complex", "very_complex"):
            matched.append(self._find_rule("SYM013"))

        return matched

    def _find_rule(self, pattern_id: str) -> SymbolicRule:
        for r in self._rules:
            if r.pattern_id == pattern_id:
                return r
        return SymbolicRule(pattern_id, "unknown", "Unknown pattern.", "info")

    def generate_symbolic_summary(self, ff: FunctionFeature, fc: Optional[FunctionContext]) -> str:
        tokens = _meaningful_tokens(ff.name)
        verb = _pick_verb(tokens)
        noun_tokens = [t for t in tokens if t not in {k for k in _VERB_MAP_KEYS}]
        noun_phrase = " ".join(noun_tokens).strip() or _humanize(ff.name)

        prefix = ""
        if ff.is_async:
            prefix = "Asynchronously "
        elif ff.is_method and ff.parent_class:
            pass

        return f"{prefix}{verb} {noun_phrase}."


_VERB_MAP_KEYS = {
    "get", "fetch", "load", "read", "set", "update", "write", "save", "store",
    "put", "add", "append", "insert", "push", "remove", "delete", "clear",
    "reset", "pop", "calc", "calculate", "compute", "count", "sum", "find",
    "search", "check", "validate", "verify", "is", "has", "can", "parse",
    "format", "convert", "transform", "encode", "decode", "process", "handle",
    "run", "execute", "start", "stop", "init", "initialize", "setup", "build",
    "create", "make", "generate", "render", "display", "show", "print", "log",
    "send", "receive", "connect", "open", "close", "sort", "filter", "map",
    "merge", "split", "join", "extract", "load_data", "test", "assert",
    "analyze", "analyse", "detect", "track", "collect", "gather", "compare",
    "evaluate", "apply", "dispatch", "emit", "publish", "subscribe", "resolve",
    "notify", "configure", "register", "unregister", "enable", "disable",
    "refresh", "clone", "copy", "move", "rename", "serialize", "deserialize",
    "dump", "restore", "backup", "import", "export", "upload", "download",
    "compress", "decompress", "hash", "sign", "authenticate", "authorize",
    "tokenize", "stem", "lemmatize", "predict", "infer", "classify", "cluster",
    "embed", "train", "fit", "score", "plot", "draw", "paint", "resize",
    "crop", "rotate", "flip",
}


def validate_consistency(
    ml_summary: str,
    ff: FunctionFeature,
    fc: Optional[FunctionContext],
    raises: List[str],
) -> List[str]:
    """
    Validate ML-generated summary against AST facts.
    Returns a list of validation flag strings describing inconsistencies.
    """
    flags: List[str] = []
    summary_lower = ml_summary.lower()

    if ff.return_annotation and ff.return_annotation not in ("None", "none"):
        ann_lower = ff.return_annotation.lower()
        if ann_lower in ("int", "float", "str", "bool", "list", "dict", "set", "tuple"):
            if ann_lower not in summary_lower and ff.return_annotation not in ml_summary:
                flags.append(
                    f"return_type_mismatch: ML summary does not mention return type "
                    f"'{ff.return_annotation}'"
                )

    if ff.loops > 0:
        loop_keywords = {"iterat", "loop", "repeat", "travers", "cycl"}
        if not any(kw in summary_lower for kw in loop_keywords):
            if "iterate" not in summary_lower and "loop" not in summary_lower:
                pass  # Not a hard error, just informational

    if raises:
        for exc in raises[:3]:
            if exc not in ml_summary and exc.lower() not in summary_lower:
                flags.append(f"missing_raise: ML summary omits raised exception '{exc}'")

    if fc and fc.security_issues:
        danger_keywords = {"danger", "security", "unsafe", "risk", "warning"}
        if not any(kw in summary_lower for kw in danger_keywords):
            flags.append(
                "missing_security_warning: function has security issues but "
                "ML summary does not mention them"
            )

    real_params = [p for p in ff.params if p.name not in ("self", "cls")]
    if real_params:
        param_names = {p.name.lower() for p in real_params}
        mentioned = sum(1 for pn in param_names if pn in summary_lower)
        if mentioned == 0 and len(real_params) > 0:
            flags.append(
                "missing_params: ML summary does not mention any of the "
                f"{len(real_params)} parameter(s)"
            )

    return flags
