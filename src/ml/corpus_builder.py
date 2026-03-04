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
        # ast.unparse on a FunctionDef already gives "def foo(args) -> ret:\n    ..."
        # We only want the first line (the signature) up to and including ":"
        unparsed = ast.unparse(node)
        sig_line = unparsed.splitlines()[0].strip()
        if not sig_line.endswith(":"):
            sig_line += ":"
        return sig_line
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
    include_codesearchnet: bool = True,
    max_stdlib_files: int = 150,
    max_files_per_pkg: int = 30,
    codesearchnet_split: str = "train",
    codesearchnet_max: int = 5000,
    deduplicate: bool = True,
    verbose: bool = True,
) -> List[CorpusEntry]:
    """
    Build a full training corpus from all available sources.

    Sources (in order):
      1. Python standard library
      2. Installed third-party packages
      3. CodeSearchNet (HuggingFace) — requires `pip install datasets`

    Returns:
        Deduplicated list of CorpusEntry objects.
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

    if include_codesearchnet:
        csn_entries = build_codesearchnet_corpus(
            split=codesearchnet_split, max_samples=codesearchnet_max, verbose=verbose
        )
        if verbose:
            print(f"  [corpus] CodeSearchNet: {len(csn_entries)} entries")
        entries.extend(csn_entries)

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


# ── CodeSearchNet dataset ─────────────────────────────────────────────────────

def build_codesearchnet_corpus(
    split: str = "train",
    max_samples: int = 5000,
    verbose: bool = True,
) -> List[CorpusEntry]:
    """
    Download and extract Python (signature, docstring) pairs from CodeSearchNet
    via Hugging Face `datasets`.

    The dataset has 412,000 Python functions with docstrings. We take a random
    sample of `max_samples` and format them as CodeT5 input/output pairs.

    Args:
        split:       Dataset split to use: "train", "validation", or "test".
        max_samples: Maximum number of entries to extract (default 5000).
        verbose:     Print progress messages.

    Returns:
        List of CorpusEntry objects.

    Raises:
        ImportError: If the `datasets` package is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        if verbose:
            print("  [corpus] datasets not installed — skipping CodeSearchNet")
        return []

    if verbose:
        print(f"  [corpus] Downloading CodeSearchNet Python ({split}) …")

    try:
        ds = load_dataset(
            "code_search_net", "python",
            split=split,
            trust_remote_code=True,
        )
    except Exception as e:
        if verbose:
            print(f"  [corpus] CodeSearchNet download failed: {e}")
        return []

    entries = []
    # Shuffle with a fixed seed for reproducibility
    import random as _rnd
    indices = list(range(len(ds)))
    _rnd.Random(42).shuffle(indices)

    for idx in indices:
        if len(entries) >= max_samples:
            break
        row = ds[idx]
        func_name  = row.get("func_name", "")       # e.g. "HttpClient.get"
        signature  = row.get("func_code_string", "") # full function source
        docstring  = row.get("func_documentation_string", "").strip()

        if not func_name or not docstring or not signature:
            continue

        # Extract simple name (last part after dot)
        simple_name = func_name.split(".")[-1]

        # Get the first line of the source (the def line) as the signature
        sig_line = signature.strip().splitlines()[0].strip()
        if not sig_line.startswith("def ") and not sig_line.startswith("async def "):
            sig_line = f"def {simple_name}():"

        # Clean the docstring
        target = _clean_docstring(docstring)
        if len(target) < 10:
            continue

        input_text = f"Summarize Python: {sig_line}"
        entries.append(CorpusEntry(
            func_name=simple_name,
            input_text=input_text,
            target_text=target,
        ))

    return entries


# ── Dataset persistence ───────────────────────────────────────────────────────

def save_corpus(
    corpus: List[CorpusEntry],
    output_dir: str,
    base_name: str = "training_corpus",
) -> dict:
    """
    Save the corpus to disk in JSON and CSV formats.

    Creates:
      {output_dir}/{base_name}.json  — full data (all fields)
      {output_dir}/{base_name}.csv   — tabular format for spreadsheet viewing

    Args:
        corpus:     List of CorpusEntry objects.
        output_dir: Directory to save files to.
        base_name:  Filename prefix (without extension).

    Returns:
        dict with file paths and corpus statistics.
    """
    import json
    import csv
    import pathlib

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{base_name}.json"
    csv_path  = out / f"{base_name}.csv"

    # JSON
    records = [
        {
            "id":          i,
            "func_name":   e.func_name,
            "input_text":  e.input_text,
            "target_text": e.target_text,
        }
        for i, e in enumerate(corpus)
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "total_samples":  len(corpus),
                "description":    "CodeT5 fine-tuning corpus: (function_signature, docstring) pairs",
                "input_format":   "Summarize Python: def func_name(params) -> return_type:",
                "target_format":  "First sentence of the function docstring.",
                "sources":        ["Python stdlib", "installed packages", "CodeSearchNet"],
            },
            "data": records,
        }, f, indent=2, ensure_ascii=False)

    # CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "func_name", "input_text", "target_text"])
        writer.writeheader()
        writer.writerows(records)

    return {
        "json_path":    str(json_path),
        "csv_path":     str(csv_path),
        "total_samples": len(corpus),
    }


def load_corpus(json_path: str) -> List[CorpusEntry]:
    """Load a saved corpus from a JSON file."""
    import json
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("data", [])
    return [
        CorpusEntry(
            func_name=r["func_name"],
            input_text=r["input_text"],
            target_text=r["target_text"],
        )
        for r in records
    ]

