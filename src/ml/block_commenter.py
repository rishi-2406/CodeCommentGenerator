"""
Block Commenter — Week 9 ML
=============================
Generates inline `# comments` above significant code blocks
(for/while/if) inside complex functions using the fine-tuned CodeT5 model.

Usage (via ml_generate_comments):
    bc = BlockCommenter(codet5_model)
    comments = bc.generate(source_code, func_feature, func_context)
    # returns List[(lineno, comment_text)]
"""
import ast
import textwrap
from typing import List, Optional, Tuple


# Minimum complexity for block commenting to activate
MIN_COMPLEXITY = 3
MIN_BODY_LINES  = 8

# Minimum confidence for a block comment to be kept
MIN_CONFIDENCE  = 0.35


class BlockCommenter:
    """
    Generates inline comments for code blocks within complex functions.

    Args:
        codet5_model: A loaded (and fine-tuned) CodeT5Model instance.
    """

    def __init__(self, codet5_model):
        self._model = codet5_model

    def generate(
        self,
        source_code: str,
        func_feature,
        func_context=None,
    ) -> List[Tuple[int, int, str]]:
        """
        Generate inline block comments for a single function.

        Args:
            source_code:  Full source code of the module (string).
            func_feature: FunctionFeature dataclass for the target function.
            func_context: FunctionContext (for complexity check), or None.

        Returns:
            List of (lineno, col_offset, comment_text) tuples.
            Returns empty list if function is simple or model is unavailable.
        """
        if self._model is None:
            return []

        # Only annotate complex functions
        complexity = getattr(func_context, "cyclomatic_complexity", 1) if func_context else 1
        body_lines  = getattr(func_feature, "body_lines", 0)
        if complexity < MIN_COMPLEXITY or body_lines < MIN_BODY_LINES:
            return []

        # Parse the source to get the function AST node
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []

        func_node = self._find_func_node(tree, func_feature.name, func_feature.lineno)
        if func_node is None:
            return []

        results: List[Tuple[int, int, str]] = []

        for node in ast.walk(func_node):
            if not isinstance(node, (ast.For, ast.While, ast.If)):
                continue
            # Skip elif branches (they share lineno with the parent if)
            if isinstance(node, ast.If) and self._is_elif(func_node, node):
                continue

            block_src = self._extract_block_source(source_code, node)
            if not block_src or len(block_src) < 15:
                continue

            # Truncate to model input length
            if len(block_src) > 250:
                block_src = block_src[:250].rsplit("\n", 1)[0]

            input_text = f"Comment Python block: {block_src}"
            try:
                comment_raw, confidence = self._model.generate(
                    input_text, num_beams=4, max_length=32
                )
            except Exception:
                continue

            if confidence < MIN_CONFIDENCE:
                continue

            # Strip docstring quotes the model may add
            comment_text = comment_raw.strip().strip('"""').strip("'''").strip()
            if not comment_text or len(comment_text) < 5:
                continue

            # Format as an inline comment
            if not comment_text.startswith("#"):
                comment_text = f"# {comment_text}"

            results.append((node.lineno, getattr(node, "col_offset", 0), comment_text))

        return results

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _find_func_node(
        tree: ast.AST, name: str, lineno: int
    ) -> Optional[ast.FunctionDef]:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == name and node.lineno == lineno:
                    return node
        return None

    @staticmethod
    def _extract_block_source(source: str, node: ast.AST) -> str:
        """Extract the source text of a block node."""
        try:
            snippet = ast.get_source_segment(source, node)
            if snippet:
                return textwrap.dedent(snippet).strip()
        except Exception:
            pass
        # Fallback: reconstruct from unparsed AST
        try:
            return ast.unparse(node).split("\n")[0].strip()
        except Exception:
            return ""

    @staticmethod
    def _is_elif(func_node: ast.FunctionDef, node: ast.If) -> bool:
        """Return True if this If node is an elif branch of another If."""
        for parent in ast.walk(func_node):
            if isinstance(parent, ast.If) and parent is not node:
                if parent.orelse and len(parent.orelse) == 1:
                    if parent.orelse[0] is node:
                        return True
        return False
