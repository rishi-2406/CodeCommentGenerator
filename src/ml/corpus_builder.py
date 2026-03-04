"""
Corpus Builder — Week 9 ML
===========================
Extracts (function_signature, docstring) pairs from:
  1. Python standard library modules (always available, offline)
  2. Selected installed third-party packages (numpy, requests, etc.)
  3. The built-in seed corpus from dataset.py

Each pair is formatted as CodeT5 input/output:
  input  → "Summarize Python: def func_name(param: type) -> ret:"
  output → "First sentence of the existing docstring."
"""
import ast
import importlib
import inspect
import os
import pkgutil
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CorpusEntry:
    func_name:  str
    input_text: str   # CodeT5 input: "Summarize Python: def ..."
    target_text: str  # CodeT5 target: clean first-line docstring


# ── Signature builder ─────────────────────────────────────────────────────────

def _build_signature(node: ast.FunctionDef) -> str:
    """Reconstruct a clean one-line function signature from an AST node."""
    try:
        return f"def {ast.unparse(node)}".split(":")[0].strip() + ":"
    except Exception:
        pass
    # Fallback: manual reconstruction
    params = []
    args = node.args
    defaults = args.defaults
    num_args = len(args.args)
    default_offset = num_args - len(defaults)
    for i, arg in enumerate(args.args):
        part = arg.arg
        if arg.annotation:
            try:
                part += f": {ast.unparse(arg.annotation)}"
            except Exception:
                pass
        if i >= default_offset:
            try:
                part += f" = {ast.unparse(defaults[i - default_offset])}"
            except Exception:
                pass
        params.append(part)
    sig = f"def {node.name}({', '.join(params)})"
    if node.returns:
        try:
            sig += f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass
    return sig + ":"


def _clean_docstring(raw: str) -> str:
    """
    Extract the first sentence/line of a docstring.
    Strips reStructuredText, NumPy-style headers, and blank filler lines.
    """
    if not raw:
        return ""
    # Dedent
    text = textwrap.dedent(raw).strip()
    # Take only the first non-empty paragraph
    first_para = text.split("\n\n")[0].strip()
    # Collapse internal newlines
    first_para = " ".join(first_para.splitlines()).strip()
    # Drop lines that look like section headers (e.g. "Parameters\n----------")
    if re.match(r'^[A-Z][a-z]+\s*$', first_para):
        return ""
    # Truncate to reasonable length
    if len(first_para) > 200:
        sentences = re.split(r'(?<=[.!?])\s+', first_para)
        first_para = sentences[0] if sentences else first_para[:200]
    # Remove trailing period if already present, then add one
    first_para = first_para.rstrip(".")
    first_para = first_para + "."
    # Final sanity check
    if len(first_para) < 6:
        return ""
    return first_para


# ── Source file parser ────────────────────────────────────────────────────────

def _parse_source_file(filepath: str, min_doc_len: int = 10) -> List[CorpusEntry]:
    """
    Parse a single Python source file and return CorpusEntry objects for every
    function that already has a docstring.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
        tree = ast.parse(src, filename=filepath)
    except (SyntaxError, OSError, UnicodeDecodeError):
        return []

    entries = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Skip private / dunder functions
        if node.name.startswith("_"):
            continue
        raw_doc = ast.get_docstring(node)
        if not raw_doc:
            continue
        target = _clean_docstring(raw_doc)
        if len(target) < min_doc_len:
            continue
        try:
            sig = _build_signature(node)
        except Exception:
            continue
        input_text = f"Summarize Python: {sig}"
        entries.append(CorpusEntry(
            func_name=node.name,
            input_text=input_text,
            target_text=target,
        ))
    return entries


# ── Standard library crawler ──────────────────────────────────────────────────

# Modules to skip (C extensions, interactive, platform-specific)
_SKIP_MODULES = {
    "antigravity", "this", "idlelib", "tkinter", "turtle",
    "test", "tests", "_thread", "nt", "posix", "winreg",
    "winsound", "msvcrt", "crypt",
}

def _stdlib_paths() -> List[str]:
    """Return a list of .py file paths from the standard library."""
    stdlib_dir = os.path.dirname(os.__file__)
    paths = []
    for root, dirs, files in os.walk(stdlib_dir):
        # Prune skip dirs
        dirs[:] = [d for d in dirs if d not in _SKIP_MODULES and not d.startswith("_")]
        for fname in files:
            if fname.endswith(".py") and not fname.startswith("_"):
                paths.append(os.path.join(root, fname))
    return paths


def build_stdlib_corpus(max_files: int = 150) -> List[CorpusEntry]:
    """
    Extract (signature, docstring) pairs from the Python standard library.

    Args:
        max_files: Maximum number of stdlib .py files to scan.

    Returns:
        List of CorpusEntry objects.
    """
    paths = _stdlib_paths()[:max_files]
    entries: List[CorpusEntry] = []
    for p in paths:
        entries.extend(_parse_source_file(p))
    return entries


# ── Installed package crawler ─────────────────────────────────────────────────

_PREFERRED_PACKAGES = [
    "requests", "flask", "django", "numpy", "pandas", "sklearn",
    "scipy", "PIL", "click", "sqlalchemy", "pydantic", "fastapi",
    "aiohttp", "boto3", "celery", "redis", "pymongo",
]

def build_package_corpus(packages: Optional[List[str]] = None,
                         max_files_per_pkg: int = 30) -> List[CorpusEntry]:
    """
    Extract (signature, docstring) pairs from installed Python packages.

    Args:
        packages:          List of package names to scan. Defaults to
                           _PREFERRED_PACKAGES (skips ones not installed).
        max_files_per_pkg: Cap files per package to avoid huge corpora.

    Returns:
        List of CorpusEntry objects.
    """
    if packages is None:
        packages = _PREFERRED_PACKAGES

    entries: List[CorpusEntry] = []
    for pkg_name in packages:
        try:
            pkg = importlib.import_module(pkg_name)
        except ImportError:
            continue
        pkg_path = getattr(pkg, "__file__", None)
        if pkg_path is None:
            continue
        pkg_dir = os.path.dirname(pkg_path)
        count = 0
        for root, dirs, files in os.walk(pkg_dir):
            dirs[:] = [d for d in dirs if not d.startswith("_")]
            for fname in files:
                if not fname.endswith(".py") or fname.startswith("_"):
                    continue
                if count >= max_files_per_pkg:
                    break
                fpath = os.path.join(root, fname)
                entries.extend(_parse_source_file(fpath))
                count += 1
    return entries


# ── Public API ────────────────────────────────────────────────────────────────

def build_full_corpus(
    include_stdlib: bool = True,
    include_packages: bool = True,
    max_stdlib_files: int = 150,
    max_files_per_pkg: int = 30,
    deduplicate: bool = True,
    verbose: bool = True,
) -> List[CorpusEntry]:
    """
    Build a full training corpus from all available sources.

    Returns:
        Deduplicated list of CorpusEntry objects sorted by function name.
    """
    entries: List[CorpusEntry] = []

    if include_stdlib:
        stdlib_entries = build_stdlib_corpus(max_files=max_stdlib_files)
        if verbose:
            print(f"  [corpus] stdlib: {len(stdlib_entries)} entries")
        entries.extend(stdlib_entries)

    if include_packages:
        pkg_entries = build_package_corpus(max_files_per_pkg=max_files_per_pkg)
        if verbose:
            print(f"  [corpus] packages: {len(pkg_entries)} entries")
        entries.extend(pkg_entries)

    if deduplicate:
        seen = set()
        unique = []
        for e in entries:
            key = (e.func_name, e.target_text[:50])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        entries = unique

    if verbose:
        print(f"  [corpus] total (deduplicated): {len(entries)} entries")
    return entries
