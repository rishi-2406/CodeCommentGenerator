"""
Comment Attacher — Week 7 Core Engine
=======================================
Inserts generated CommentItem objects back into the original source code.

Strategy:
  - For "docstring" kind  → insert the docstring as the first line of the
    function/class body (right after the def/class line)
  - For "inline" / "block" kind → insert as a comment line ABOVE the node

All original indentation is preserved. The result is returned as a string
along with a simple diff log.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

from .comment_generator import CommentItem


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AttachResult:
    annotated_source: str
    diff_log: List[str] = field(default_factory=list)   # human-readable diff lines
    comments_attached: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _indent(col_offset: int) -> str:
    """Return a string of spaces matching the node's indentation."""
    return " " * col_offset


def _indent_block(text: str, col_offset: int) -> str:
    """Indent every line of a multi-line text block."""
    pad = _indent(col_offset + 4)  # +4 for inside the def body
    return "\n".join(pad + line for line in text.splitlines())


def _format_docstring_lines(raw_docstring: str, col_offset: int) -> List[str]:
    """
    Convert a raw docstring (with triple-quotes) into indented source lines.
    The docstring goes inside the function body, so it's indented by
    col_offset + 4 spaces (one level deeper than the def).
    """
    inner_indent = _indent(col_offset + 4)
    lines = raw_docstring.splitlines()
    result = []
    for line in lines:
        result.append(inner_indent + line)
    return result


def _format_inline_lines(raw_comment: str, col_offset: int) -> List[str]:
    """
    Split a raw inline/block comment into indented source lines,
    placed at the same indent level as the def/class statement.
    """
    pad = _indent(col_offset)
    return [pad + line for line in raw_comment.splitlines()]


# ---------------------------------------------------------------------------
# Main attacher
# ---------------------------------------------------------------------------

def attach_comments(source_code: str, comments: List[CommentItem]) -> AttachResult:
    """
    Insert CommentItems into source_code and return the annotated result.

    Args:
        source_code: Original Python source string.
        comments:    List of CommentItem (sorted by lineno, output of generator).

    Returns:
        AttachResult with the annotated source and a diff log.
    """
    source_lines = source_code.splitlines()
    # We'll build an output list of (original_lineno, text) to allow
    # easy insertion without off-by-one errors.
    # Use a dict: original_0indexed_line -> list of lines to insert BEFORE it.
    insertions: dict = {}   # {0-indexed line : [strings to insert before]}

    diff_log: List[str] = []
    attached = 0

    for item in comments:
        # lineno is 1-indexed (from AST), convert to 0-indexed
        line_idx = item.lineno - 1

        if item.kind == "docstring":
            # Insert AFTER the def line → that is, before line_idx+1
            insert_at = line_idx + 1
            new_lines = _format_docstring_lines(item.text, item.col_offset)
        else:
            # Insert BEFORE the def/class line
            insert_at = line_idx
            new_lines = _format_inline_lines(item.text, item.col_offset)

        insertions.setdefault(insert_at, []).extend(new_lines)
        diff_log.append(
            f"  + [{item.kind}] {item.node_type} '{item.target_name}' "
            f"(before original line {insert_at + 1}):"
        )
        for nl in new_lines:
            diff_log.append(f"      {nl}")
        attached += 1

    # Rebuild source with insertions
    output_lines: List[str] = []
    for i, original_line in enumerate(source_lines):
        if i in insertions:
            output_lines.extend(insertions[i])
        output_lines.append(original_line)

    # Handle insertions beyond the last line (edge case)
    if len(source_lines) in insertions:
        output_lines.extend(insertions[len(source_lines)])

    return AttachResult(
        annotated_source="\n".join(output_lines),
        diff_log=diff_log,
        comments_attached=attached,
    )
