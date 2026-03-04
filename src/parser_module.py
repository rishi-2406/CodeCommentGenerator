"""
Parser module: reads Python source files and builds an AST.
Re-exported from week6 with minor enhancements for week7 pipeline.
"""
import ast
from .error_handler import ParserError


def read_file(filepath: str) -> str:
    """Reads the content of a source file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise ParserError(f"File not found: {filepath}")
    except Exception as e:
        raise ParserError(f"Error reading file: {str(e)}")


def parse_code(source_code: str) -> ast.AST:
    """
    Parses Python source code into an AST.

    Returns the root Module node on success.
    Raises ParserError on syntax errors.
    """
    try:
        tree = ast.parse(source_code)
        return tree
    except SyntaxError as e:
        raise ParserError(f"Syntax error: {e.msg}", line=e.lineno, column=e.offset)
    except Exception as e:
        raise ParserError(f"Unexpected error during parsing: {str(e)}")
