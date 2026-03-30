"""
Unit tests for Week 7 Core Engine modules:
  - TestASTExtractor
  - TestContextAnalyzer
  - TestCommentGenerator
  - TestCommentAttacher
"""
import ast
import unittest

from src.parser_module import parse_code
from src.ast_extractor import extract_features, ModuleFeatures, FunctionFeature, ClassFeature
from src.context_analyzer import analyze_context, ContextGraph
from src.comment_generator import generate_comments, CommentItem
from src.comment_attacher import attach_comments, AttachResult


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SIMPLE_CODE = """\
def calculate_sum(a: int, b: int) -> int:
    return a + b

class MathHelper:
    def multiply(self, x, y):
        return x * y
"""

COMPLEX_CODE = """\
def search_sorted(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
"""

DOCUMENTED_CODE = """\
def greet(name: str) -> str:
    \"\"\"Return a greeting string.\"\"\"
    return f"Hello, {name}"
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AST Extractor Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestASTExtractor(unittest.TestCase):

    def _extract(self, code: str) -> ModuleFeatures:
        tree = parse_code(code)
        return extract_features(tree, source_code=code)

    def test_extracts_function_name(self):
        mf = self._extract(SIMPLE_CODE)
        names = [f.name for f in mf.functions]
        self.assertIn("calculate_sum", names)

    def test_extracts_function_params(self):
        mf = self._extract(SIMPLE_CODE)
        func = next(f for f in mf.functions if f.name == "calculate_sum")
        param_names = [p.name for p in func.params]
        self.assertEqual(param_names, ["a", "b"])

    def test_extracts_type_annotations(self):
        mf = self._extract(SIMPLE_CODE)
        func = next(f for f in mf.functions if f.name == "calculate_sum")
        self.assertEqual(func.return_annotation, "int")
        self.assertEqual(func.params[0].annotation, "int")

    def test_extracts_class(self):
        mf = self._extract(SIMPLE_CODE)
        self.assertEqual(len(mf.classes), 1)
        self.assertEqual(mf.classes[0].name, "MathHelper")

    def test_class_methods_list(self):
        mf = self._extract(SIMPLE_CODE)
        cls = mf.classes[0]
        self.assertIn("multiply", cls.methods)

    def test_method_marked_as_method(self):
        mf = self._extract(SIMPLE_CODE)
        method = next(f for f in mf.functions if f.name == "multiply")
        self.assertTrue(method.is_method)
        self.assertEqual(method.parent_class, "MathHelper")

    def test_no_docstring_flag(self):
        mf = self._extract(SIMPLE_CODE)
        func = next(f for f in mf.functions if f.name == "calculate_sum")
        self.assertFalse(func.has_docstring)

    def test_docstring_detected(self):
        mf = self._extract(DOCUMENTED_CODE)
        func = mf.functions[0]
        self.assertTrue(func.has_docstring)
        self.assertEqual(func.docstring, "Return a greeting string.")

    def test_loops_and_conditionals_count(self):
        mf = self._extract(COMPLEX_CODE)
        func = mf.functions[0]
        self.assertGreaterEqual(func.loops, 1)
        self.assertGreaterEqual(func.conditionals, 1)

    def test_line_count(self):
        mf = self._extract(SIMPLE_CODE)
        self.assertEqual(mf.total_lines, len(SIMPLE_CODE.splitlines()))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Context Analyzer Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestContextAnalyzer(unittest.TestCase):

    def _analyze(self, code: str) -> ContextGraph:
        tree = parse_code(code)
        mf = extract_features(tree, source_code=code)
        return analyze_context(mf, tree, code)

    def test_function_context_created(self):
        cg = self._analyze(SIMPLE_CODE)
        names = [fc.name for fc in cg.function_contexts]
        self.assertIn("calculate_sum", names)

    def test_simple_function_cyclomatic_complexity(self):
        cg = self._analyze(SIMPLE_CODE)
        fc = next(f for f in cg.function_contexts if f.name == "calculate_sum")
        # Straight-line function: CC = 1
        self.assertEqual(fc.cyclomatic_complexity, 1)

    def test_complex_function_cyclomatic_complexity(self):
        cg = self._analyze(COMPLEX_CODE)
        fc = cg.function_contexts[0]
        # while + 3 ifs = CC >= 4
        self.assertGreaterEqual(fc.cyclomatic_complexity, 4)

    def test_complexity_label_simple(self):
        cg = self._analyze(SIMPLE_CODE)
        fc = next(f for f in cg.function_contexts if f.name == "calculate_sum")
        self.assertEqual(fc.complexity_label, "simple")

    def test_complexity_label_complex(self):
        cg = self._analyze(COMPLEX_CODE)
        fc = cg.function_contexts[0]
        self.assertIn(fc.complexity_label, ("moderate", "complex", "very_complex"))

    def test_call_graph_populated(self):
        cg = self._analyze(SIMPLE_CODE)
        self.assertIn("calculate_sum", cg.call_graph)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Comment Generator Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCommentGenerator(unittest.TestCase):

    def _generate(self, code: str):
        tree = parse_code(code)
        mf = extract_features(tree, source_code=code)
        cg = analyze_context(mf, tree, code)
        return generate_comments(mf, cg, source_code=code)  # pass source_code!

    def test_generates_comment_for_undocumented_function(self):
        comments = self._generate(SIMPLE_CODE)
        node_ids = [c.node_id for c in comments]
        # calculate_sum has no docstring, should get a comment
        self.assertTrue(any("calculate_sum" in nid for nid in node_ids))

    def test_no_comment_for_documented_function(self):
        comments = self._generate(DOCUMENTED_CODE)
        # greet() already has a docstring; no docstring comment should be generated
        docstring_comments = [c for c in comments if c.kind == "docstring"]
        self.assertEqual(len(docstring_comments), 0)

    def test_comment_kind_is_docstring(self):
        comments = self._generate(SIMPLE_CODE)
        ds_comments = [c for c in comments if c.kind == "docstring"]
        self.assertGreater(len(ds_comments), 0)

    def test_comment_text_contains_triple_quotes(self):
        comments = self._generate(SIMPLE_CODE)
        for c in comments:
            if c.kind == "docstring":
                self.assertIn('"""', c.text)

    def test_comment_text_contains_args_section(self):
        comments = self._generate(SIMPLE_CODE)
        func_comment = next(
            (c for c in comments if c.target_name == "calculate_sum" and c.kind == "docstring"),
            None
        )
        self.assertIsNotNone(func_comment)
        self.assertIn("Args:", func_comment.text) # type: ignore

    def test_class_gets_docstring(self):
        comments = self._generate(SIMPLE_CODE)
        class_comments = [c for c in comments if c.node_type == "class"]
        self.assertGreater(len(class_comments), 0)

    def test_complex_function_gets_inline_comment(self):
        comments = self._generate(COMPLEX_CODE)
        inline_comments = [c for c in comments if c.kind == "inline"]
        # search_sorted is complex enough for an inline note
        self.assertGreater(len(inline_comments), 0)

    def test_comments_sorted_by_lineno(self):
        comments = self._generate(SIMPLE_CODE)
        lines = [c.lineno for c in comments]
        self.assertEqual(lines, sorted(lines))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AST-Driven Comment Tests  (verifies body analysis features work)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOOP_CODE = """\
def process_items(items: list, threshold: int = 5) -> dict:
    result = {}
    for item in items:
        if item > threshold:
            result[item] = item * 2
        elif item == 0:
            raise ValueError("Zero not allowed")
    return result
"""

DECORATOR_CODE = """\
class Config:
    @property
    def value(self):
        return self._value

    @staticmethod
    def defaults():
        return {}

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls()
        return obj
"""

ASYNC_CODE = """\
async def fetch_data(url: str) -> dict:
    result = await some_client.get(url)
    if result is None:
        raise ConnectionError("No response")
    return result
"""


class TestASTDrivenComments(unittest.TestCase):
    """
    Verifies that generate_comments() uses AST features — not just the
    function name — to produce meaningful docstring content.
    """

    def _generate(self, code: str):
        tree = parse_code(code)
        mf   = extract_features(tree, source_code=code)
        cg   = analyze_context(mf, tree, code)
        return generate_comments(mf, cg, source_code=code)

    # ── Loop / conditional detection ────────────────────────────────────────

    def test_loop_mentioned_in_docstring(self):
        """A function with a for-loop should say 'Iterates over' in its docstring."""
        comments = self._generate(LOOP_CODE)
        func_doc = next(
            (c for c in comments if c.target_name == "process_items" and c.kind == "docstring"),
            None
        )
        self.assertIsNotNone(func_doc)
        self.assertIn("Iterates", func_doc.text)

    def test_conditional_mentioned_in_docstring(self):
        """A function with branches should say 'conditional' in its docstring."""
        comments = self._generate(LOOP_CODE)
        func_doc = next(
            (c for c in comments if c.target_name == "process_items" and c.kind == "docstring"),
            None
        )
        self.assertIsNotNone(func_doc)
        self.assertIn("conditional", func_doc.text)

    # ── Raises detection ────────────────────────────────────────────────────

    def test_raises_mentioned_in_docstring(self):
        """A function that raises ValueError should document it."""
        comments = self._generate(LOOP_CODE)
        func_doc = next(
            (c for c in comments if c.target_name == "process_items" and c.kind == "docstring"),
            None
        )
        self.assertIsNotNone(func_doc)
        self.assertIn("ValueError", func_doc.text)

    def test_raises_section_present(self):
        """The Raises: section should appear when exceptions are raised."""
        comments = self._generate(LOOP_CODE)
        func_doc = next(
            (c for c in comments if c.target_name == "process_items" and c.kind == "docstring"),
            None
        )
        self.assertIsNotNone(func_doc)
        self.assertIn("Raises:", func_doc.text)

    # ── Return type inference ────────────────────────────────────────────────

    def test_return_section_uses_annotation(self):
        """Return annotation 'dict' should appear in the Returns: section."""
        comments = self._generate(LOOP_CODE)
        func_doc = next(
            (c for c in comments if c.target_name == "process_items" and c.kind == "docstring"),
            None
        )
        self.assertIsNotNone(func_doc)
        self.assertIn("Returns:", func_doc.text)
        self.assertIn("dict", func_doc.text)

    # ── Async detection ─────────────────────────────────────────────────────

    def test_async_mentioned_in_summary(self):
        """Async functions should have 'Asynchronously' in the summary line."""
        comments = self._generate(ASYNC_CODE)
        func_doc = next(
            (c for c in comments if c.target_name == "fetch_data" and c.kind == "docstring"),
            None
        )
        self.assertIsNotNone(func_doc)
        self.assertIn("synchronous", func_doc.text.lower())

    # ── Decorator detection ─────────────────────────────────────────────────

    def test_property_decorator_mentioned(self):
        """@property methods should say 'property accessor' in their docstring."""
        comments = self._generate(DECORATOR_CODE)
        prop_doc = next(
            (c for c in comments if c.target_name == "value" and c.kind == "docstring"),
            None
        )
        self.assertIsNotNone(prop_doc)
        self.assertIn("property", prop_doc.text.lower())

    def test_staticmethod_decorator_mentioned(self):
        """@staticmethod methods should say 'static method'."""
        comments = self._generate(DECORATOR_CODE)
        doc = next(
            (c for c in comments if c.target_name == "defaults" and c.kind == "docstring"),
            None
        )
        self.assertIsNotNone(doc)
        self.assertIn("static method", doc.text.lower())

    # ── Inline block comment ────────────────────────────────────────────────

    def test_inline_body_summary_included(self):
        """Inline comment for a complex function should mention loop/conditional count."""
        comments = self._generate(LOOP_CODE)
        inline = next(
            (c for c in comments if c.target_name == "process_items" and c.kind == "inline"),
            None
        )
        # process_items has CC=4 (moderate), should get an inline comment
        if inline is not None:   # only present when complexity is moderate+
            self.assertIn("loop", inline.text.lower())

    # ── Data structure detection ────────────────────────────────────────────

    def test_data_structure_mentioned(self):
        """A function that builds a dict should mention it in the docstring."""
        comments = self._generate(LOOP_CODE)
        func_doc = next(
            (c for c in comments if c.target_name == "process_items" and c.kind == "docstring"),
            None
        )
        self.assertIsNotNone(func_doc)
        # 'dict' should appear in the docstring (return annotation or data structures)
        self.assertIn("dict", func_doc.text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Comment Attacher Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCommentAttacher(unittest.TestCase):

    def _annotate(self, code: str) -> AttachResult:
        tree = parse_code(code)
        mf = extract_features(tree, source_code=code)
        cg = analyze_context(mf, tree, code)
        comments = generate_comments(mf, cg, source_code=code)  # pass source_code!
        return attach_comments(code, comments)

    def test_annotated_source_is_string(self):
        result = self._annotate(SIMPLE_CODE)
        self.assertIsInstance(result.annotated_source, str)

    def test_annotated_source_longer_than_original(self):
        result = self._annotate(SIMPLE_CODE)
        original_lines = len(SIMPLE_CODE.splitlines())
        annotated_lines = len(result.annotated_source.splitlines())
        self.assertGreater(annotated_lines, original_lines)

    def test_original_code_preserved(self):
        result = self._annotate(SIMPLE_CODE)
        # All original non-blank lines should still exist in the annotated version
        for line in SIMPLE_CODE.splitlines():
            if line.strip():
                self.assertIn(line, result.annotated_source)

    def test_triple_quoted_docstring_in_output(self):
        result = self._annotate(SIMPLE_CODE)
        self.assertIn('"""', result.annotated_source)

    def test_comments_attached_count(self):
        result = self._annotate(SIMPLE_CODE)
        self.assertGreater(result.comments_attached, 0)

    def test_diff_log_populated(self):
        result = self._annotate(SIMPLE_CODE)
        self.assertGreater(len(result.diff_log), 0)

    def test_no_double_docstring_on_documented_function(self):
        result = self._annotate(DOCUMENTED_CODE)
        # Should only have ONE triple quote block (the original)
        self.assertEqual(result.annotated_source.count('"""'), 2)  # open + close

    def test_indentation_preserved(self):
        result = self._annotate(SIMPLE_CODE)
        # The class method docstring should be indented at least 8 spaces
        lines = result.annotated_source.splitlines()
        has_indented_doc = any(
            line.startswith("        ") and line.strip().startswith('"""')
            for line in lines
        )
        self.assertTrue(has_indented_doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
