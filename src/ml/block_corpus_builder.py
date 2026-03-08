"""
Block Corpus Builder — Week 9 ML
==================================
Streams Python files from `bigcode/the-stack-dedup` and extracts
(code_block, inline_comment) pairs.

Extraction rule:
  A `# comment` on the line IMMEDIATELY above a `for`, `while`, or `if`
  statement is treated as the description of that code block.

Example mined pair:
  input  → "Comment Python block: while low <= high:\\n    mid = ..."
  target → "Binary search: repeatedly halves the search space."

Usage:
  corpus = build_block_corpus(max_files=8000, max_pairs=2500)
  save_block_corpus(corpus, "outputs/block_corpus.json")
"""
import ast
import re
import textwrap
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BlockEntry:
    block_type: str          # "for" | "while" | "if"
    input_text: str          # "Comment Python block: <code>"
    target_text: str         # inline comment text (cleaned)
    source_file: str = ""    # optional provenance


# ── Block extraction from a single source string ──────────────────────────────

_BLOCK_START = re.compile(r"^\s*(for|while|if)\b")
_COMMENT_LINE = re.compile(r"^\s*#\s*(.*)")


def _clean_comment(raw: str) -> str:
    """Remove leading # and whitespace, ensure sentence ends with a period."""
    text = raw.strip().lstrip("#").strip()
    if not text or len(text) < 5:
        return ""
    # Ignore pure separator lines like "# ---"
    if re.match(r"^[-=*#\s]+$", text):
        return ""
    # Ignore shebang / encoding declarations
    if text.startswith("!") or "coding" in text[:20]:
        return ""
    if not text.endswith("."):
        text += "."
    return text


def _block_source(lines: List[str], start_idx: int, indent: int,
                  max_lines: int = 12) -> str:
    """Extract source lines of a block starting at start_idx."""
    block_lines = [lines[start_idx]]
    for i in range(start_idx + 1, min(start_idx + max_lines, len(lines))):
        line = lines[i]
        # Stop if we've left the block (dedented back to or past indent)
        if line.strip() == "":
            block_lines.append("")
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= indent and line.strip():
            break
        block_lines.append(line.rstrip())
    return "\n".join(block_lines).strip()


def extract_block_pairs(source: str, filepath: str = "") -> List[BlockEntry]:
    """
    Extract (block_code, inline_comment) pairs from a Python source string.

    Args:
        source:   Python source code as a string.
        filepath: Optional filename for provenance.

    Returns:
        List of BlockEntry objects.
    """
    # Quick sanity check — must be parseable
    try:
        ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    entries = []

    for i, line in enumerate(lines):
        # Check if previous line is a standalone comment
        if i == 0:
            continue
        prev = lines[i - 1]
        prev_m = _COMMENT_LINE.match(prev)
        if not prev_m:
            continue

        # Check if current line is a block start
        cur_m = _BLOCK_START.match(line)
        if not cur_m:
            continue

        comment_text = _clean_comment(prev_m.group(1))
        if not comment_text:
            continue

        block_type = cur_m.group(1)
        indent = len(line) - len(line.lstrip())
        block_src = _block_source(lines, i, indent)

        if len(block_src) < 15:
            continue

        # Truncate block to ~200 chars for model input
        if len(block_src) > 250:
            block_src = block_src[:250].rsplit("\n", 1)[0]

        input_text = f"Comment Python block: {block_src}"
        entries.append(BlockEntry(
            block_type=block_type,
            input_text=input_text,
            target_text=comment_text,
            source_file=filepath,
        ))

    return entries


# ── Dataset builder — streams from the-stack and alpaca ───────────

def build_instructional_block_corpus(max_pairs: int = 5000, verbose: bool = True) -> List[BlockEntry]:
    """Stream from tarun's python alpaca instructions to build high quality block explanations."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []
        
    if verbose:
        print("  [block] Streaming Alpaca/Instructional Python blocks...")
        
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
    except Exception as e:
        if verbose: print(f"  [block] Alpaca dataset skipped: {e}")
        return []

    entries: List[BlockEntry] = []
    seen = set()
    for row in ds:
        code = row.get("output", "").strip()
        inst = row.get("instruction", "").strip()
        if not code or not inst or len(code) < 15 or len(code) > 250:
            continue
            
        # Clean up prompts
        clean_inst = re.sub(r'^(Create|Write|Implement|Generate|Develop)\b\s*a?\s*(Python|python|script|program|function|application)?\b\s*(to|that)?\s*', '', inst, flags=re.IGNORECASE).strip()
        if not clean_inst:
            continue
        clean_inst = clean_inst[0].upper() + clean_inst[1:]
        if not clean_inst.endswith("."): clean_inst += "."
        
        input_text = f"Comment Python block: {code}"
        key = (input_text[:80], clean_inst[:40])
        if key not in seen:
            seen.add(key)
            entries.append(BlockEntry(
                block_type="alpaca",
                input_text=input_text,
                target_text=clean_inst,
                source_file="alpaca"
            ))
            if len(entries) >= max_pairs:
                break
    return entries



def build_block_corpus(
    max_files: int = 20000,
    max_pairs: int = 7500,
    verbose: bool = True,
) -> List[BlockEntry]:
    """
    Stream Python files from bigcode/the-stack-dedup and extract
    (code_block, inline_comment) pairs.

    Args:
        max_files:  Maximum number of Python files to scan (streaming).
        max_pairs:  Stop after collecting this many pairs.
        verbose:    Print progress.

    Returns:
        List of BlockEntry objects (deduplicated).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        if verbose:
            print("  [block] `datasets` not installed — run: pip install datasets")
        return []

    if verbose:
        print(f"  [block] Streaming the-stack-dedup Python (target: {max_pairs} pairs) …")

    entries: List[BlockEntry] = []
    
    try:
        ds = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        if verbose:
            print(f"  [block] the-stack-dedup unavailable: {e}")
            print("  [block] Falling back to local stdlib block extraction …")
        ds = []
        entries.extend(_build_stdlib_block_corpus(max_pairs=max_pairs, verbose=verbose))

    # Pre-emptively append the Alpaca instructions dataset as highly relevant block components
    alpaca_entries = build_instructional_block_corpus(max_pairs=max_pairs // 2, verbose=verbose)
    if alpaca_entries:
        if verbose: print(f"  [block] Alpaca provided {len(alpaca_entries)} highly relevant blocks.")
        entries.extend(alpaca_entries)

    seen = { (e.input_text[:80], e.target_text[:40]) for e in entries }
    files_scanned = 0

    for row in ds:
        if len(entries) >= max_pairs or files_scanned >= max_files:
            break
        files_scanned += 1
        content = row.get("content", "")
        if not content or len(content) > 100_000:
            continue
        new_entries = extract_block_pairs(content)
        for e in new_entries:
            key = (e.input_text[:80], e.target_text[:40])
            if key not in seen:
                seen.add(key)
                entries.append(e)
                if len(entries) >= max_pairs:
                    break

        if verbose and files_scanned % 1000 == 0:
            print(f"  [block] Scanned {files_scanned} files → {len(entries)} pairs")

    if verbose:
        print(f"  [block] Done: {len(entries)} block pairs from {files_scanned} files")
    return entries


def _build_stdlib_block_corpus(max_pairs: int = 2500, verbose: bool = True) -> List[BlockEntry]:
    """Fallback: mine block comment pairs from Python stdlib."""
    import os
    stdlib_dir = os.path.dirname(os.__file__)
    entries: List[BlockEntry] = []
    seen = set()

    for root, dirs, files in os.walk(stdlib_dir):
        dirs[:] = [d for d in dirs if not d.startswith("_") and d not in {"test", "tests"}]
        for fname in files:
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    src = f.read()
            except OSError:
                continue
            for e in extract_block_pairs(src, fpath):
                key = (e.input_text[:80], e.target_text[:40])
                if key not in seen:
                    seen.add(key)
                    entries.append(e)
            if len(entries) >= max_pairs:
                break
        if len(entries) >= max_pairs:
            break

    if verbose:
        print(f"  [block] stdlib fallback: {len(entries)} block pairs")
    return entries


# ── Persistence ───────────────────────────────────────────────────────────────

def save_block_corpus(corpus: List[BlockEntry], output_dir: str,
                      base_name: str = "block_corpus") -> dict:
    """Save block corpus as JSON and CSV."""
    import json
    import csv
    import pathlib

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{base_name}.json"
    csv_path  = out / f"{base_name}.csv"

    records = [
        {
            "id":         i,
            "block_type": e.block_type,
            "input_text": e.input_text,
            "target_text": e.target_text,
        }
        for i, e in enumerate(corpus)
    ]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "total_samples": len(corpus),
                "description": "CodeT5 block-comment corpus: (code_block → inline_comment) pairs",
                "input_format": "Comment Python block: <code>",
                "target_format": "Inline comment describing what the block does.",
                "source": "bigcode/the-stack-dedup (Python)",
            },
            "data": records,
        }, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "block_type", "input_text", "target_text"])
        w.writeheader()
        w.writerows(records)

    return {"json_path": str(json_path), "csv_path": str(csv_path),
            "total_samples": len(corpus)}
