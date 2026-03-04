"""
Semantic validator: checks AST for style and structural issues.
Carried over from week6, ensures pipeline correctness before extraction.
"""
import ast
from typing import List
from .error_handler import ParserError


class SemanticValidator(ast.NodeVisitor):
    """Visits AST nodes and collects style/semantic violations."""

    def __init__(self):
        self.errors: List[ParserError] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Enforce snake_case function names
        if not node.name.islower():
            self.errors.append(ParserError(
                f"Function '{node.name}' should be snake_case.",
                line=node.lineno,
                column=node.col_offset
            ))
        # Flag overly complex signatures
        if len(node.args.args) > 10:
            self.errors.append(ParserError(
                f"Function '{node.name}' has too many arguments ({len(node.args.args)} > 10).",
                line=node.lineno,
                column=node.col_offset
            ))
        self.generic_visit(node)

    # Also validate async functions with the same rules
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef):
        # Enforce CamelCase class names
        if not node.name[0].isupper():
            self.errors.append(ParserError(
                f"Class '{node.name}' should be CamelCase (start with uppercase).",
                line=node.lineno,
                column=node.col_offset
            ))
        self.generic_visit(node)


def validate_ast(tree: ast.AST) -> List[ParserError]:
    """Validates the AST for semantic rules. Returns a list of errors."""
    validator = SemanticValidator()
    validator.visit(tree)
    return validator.errors
