"""
AST Body Extractor — AST-Driven Comment Generation
====================================================
Provides utilities to extract a clean, concise function-body snippet
from raw source code using line-number information already stored in
FunctionFeature.  The snippet is used by:

  1. The rule-based engine — to enrich generated comments with
     natural-language descriptions of what the body actually does.
  2. The ML / CodeT5 path — as additional input context so the model
     can generate comments that describe the *implementation*, not
     just the name.
"""
import ast
import textwrap
from typing import List, Optional


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def extract_body_snippet(
    source_code: str,
    lineno: int,
    end_lineno: int,
    max_lines: int = 15,
) -> str:
    """
    Extract the function body from source_code as a clean text snippet.

    Strips the leading docstring (if any) from the body so the ML model
    is not trained / prompted with an existing comment.  Limits output
    to *max_lines* lines and dedents for readability.

    Args:
        source_code: Full source text of the module.
        lineno:      First line of the function definition (1-indexed).
        end_lineno:  Last line of the function definition (1-indexed).
        max_lines:   Maximum body lines to include in the snippet.

    Returns:
        Dedented body snippet string (may be empty if extraction fails).
    """
    lines = source_code.splitlines()
    # Slice out the function block (lineno / end_lineno are 1-indexed)
    func_lines = lines[lineno - 1 : end_lineno]
    if not func_lines:
        return ""

    func_src = "\n".join(func_lines)
    try:
        tree = ast.parse(textwrap.dedent(func_src))
    except SyntaxError:
        # Fall back to raw line slice if re-parse fails
        body_lines = func_lines[1:]  # drop the def line
        body_lines = _strip_docstring_lines(body_lines)
        return _limit_and_dedent(body_lines, max_lines)

    # Find the first FunctionDef / AsyncFunctionDef in the mini-tree
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = node
            break

    if func_node is None:
        return ""

    # Remove docstring node from body statements
    body_stmts = _drop_docstring_stmt(func_node.body)
    if not body_stmts:
        return ""

    # Unparse the remaining statements to clean Python source
    try:
        snippet_lines: List[str] = []
        for stmt in body_stmts:
            unparsed = ast.unparse(stmt)
            snippet_lines.extend(unparsed.splitlines())
    except Exception:
        # Fallback: use raw slice
        raw = func_lines[1:]
        raw = _strip_docstring_lines(raw)
        return _limit_and_dedent(raw, max_lines)

    return "\n".join(snippet_lines[:max_lines])


def extract_raises(source_code: str, lineno: int, end_lineno: int) -> List[str]:
    """
    Return a list of exception types raised explicitly inside a function body.

    Args:
        source_code: Full module source.
        lineno:      First line of the function (1-indexed).
        end_lineno:  Last line of the function (1-indexed).

    Returns:
        Deduplicated list of exception name strings, e.g. ["ValueError", "IOError"].
    """
    lines = source_code.splitlines()
    func_src = "\n".join(lines[lineno - 1 : end_lineno])
    try:
        tree = ast.parse(textwrap.dedent(func_src))
    except SyntaxError:
        return []

    raises: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            try:
                exc_str = ast.unparse(node.exc)
                # Keep just the exception class name (strip call parens)
                exc_name = exc_str.split("(")[0].strip()
                if exc_name and exc_name not in raises:
                    raises.append(exc_name)
            except Exception:
                pass
    return raises


def extract_returned_types(source_code: str, lineno: int, end_lineno: int) -> List[str]:
    """
    Return a best-effort list of types/values that a function returns.

    Inspects every ``return`` statement in the body and infers the type
    from the returned expression (constants, collections, calls, names).

    Args:
        source_code: Full module source.
        lineno:      First line of the function (1-indexed).
        end_lineno:  Last line of the function (1-indexed).

    Returns:
        Deduplicated list of inferred return-type strings.
    """
    lines = source_code.splitlines()
    func_src = "\n".join(lines[lineno - 1 : end_lineno])
    try:
        tree = ast.parse(textwrap.dedent(func_src))
    except SyntaxError:
        return []

    types: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            t = _infer_return_type(node.value)
            if t and t not in types:
                types.append(t)
    return types


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _drop_docstring_stmt(body: list) -> list:
    """Remove the leading docstring Expr node from an AST body list."""
    if not body:
        return body
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return body[1:]
    return body


def _strip_docstring_lines(lines: List[str]) -> List[str]:
    """
    Strip a triple-quoted docstring from the top of a list of lines
    (raw text fallback when AST re-parse fails).
    """
    stripped = [l.strip() for l in lines]
    if not stripped:
        return lines
    if stripped[0].startswith('"""') or stripped[0].startswith("'''"):
        quote = stripped[0][:3]
        # Single-line docstring
        if stripped[0].count(quote) >= 2 and len(stripped[0]) > 3:
            return lines[1:]
        # Multi-line: find closing triple-quote
        for i, s in enumerate(stripped[1:], 1):
            if quote in s:
                return lines[i + 1 :]
    return lines


def _limit_and_dedent(lines: List[str], max_lines: int) -> str:
    """Dedent and cap lines."""
    limited = lines[:max_lines]
    return textwrap.dedent("\n".join(limited)).strip()


def _infer_return_type(node: ast.expr) -> Optional[str]:
    """
    Heuristically infer a human-readable type label from a return expression.
    """
    if isinstance(node, ast.Constant):
        t = type(node.value).__name__
        return t if t != "NoneType" else "None"
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.Set):
        return "set"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.Call):
        try:
            return ast.unparse(node.func).split(".")[-1]
        except Exception:
            return "object"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BoolOp):
        return "bool"
    if isinstance(node, ast.Compare):
        return "bool"
    return None
