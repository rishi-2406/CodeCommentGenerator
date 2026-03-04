"""
Basic parser and validator tests (carried over from week6).
"""
import ast
import unittest

from src.parser_module import parse_code
from src.error_handler import ParserError
from src.validator import validate_ast


class TestParser(unittest.TestCase):

    def test_valid_code(self):
        code = "def foo(): pass"
        tree = parse_code(code)
        self.assertIsInstance(tree, ast.AST)

    def test_syntax_error(self):
        code = "def foo() pass"  # Missing colon
        with self.assertRaises(ParserError) as cm:
            parse_code(code)
        self.assertIn("Syntax error", str(cm.exception))

    def test_empty_module(self):
        tree = parse_code("")
        self.assertIsInstance(tree, ast.Module)

    def test_complex_code_parses(self):
        code = """
import os
from typing import List

class Foo:
    def bar(self, x: int) -> str:
        return str(x)
"""
        tree = parse_code(code)
        self.assertIsInstance(tree, ast.AST)


class TestValidator(unittest.TestCase):

    def test_valid_snake_case_function(self):
        code = "def good_function(): pass\nclass GoodClass: pass"
        tree = parse_code(code)
        errors = validate_ast(tree)
        self.assertEqual(len(errors), 0)

    def test_invalid_function_naming(self):
        code = "def BadFunction(): pass"
        tree = parse_code(code)
        errors = validate_ast(tree)
        self.assertEqual(len(errors), 1)
        self.assertIn("snake_case", errors[0].message)

    def test_invalid_class_naming(self):
        code = "class badClass: pass"
        tree = parse_code(code)
        errors = validate_ast(tree)
        self.assertEqual(len(errors), 1)
        self.assertIn("CamelCase", errors[0].message)

    def test_too_many_arguments(self):
        args = ", ".join(f"a{i}" for i in range(11))
        code = f"def huge_func({args}): pass"
        tree = parse_code(code)
        errors = validate_ast(tree)
        self.assertTrue(any("too many arguments" in e.message for e in errors))

    def test_multiple_violations(self):
        code = """
def BadFunction():
    pass
class badClass:
    pass
"""
        tree = parse_code(code)
        errors = validate_ast(tree)
        self.assertEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
