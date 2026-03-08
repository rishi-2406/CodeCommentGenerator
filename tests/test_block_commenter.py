"""
Unit Tests — Block Commenter (Week 9)
======================================
Tests for block_corpus_builder and BlockCommenter.
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ml.block_corpus_builder import (
    extract_block_pairs, BlockEntry, _clean_comment, save_block_corpus
)

# ── _clean_comment ────────────────────────────────────────────────────────────

class TestCleanComment:
    def test_basic(self):
        assert _clean_comment("# Binary search") == "Binary search."

    def test_adds_period(self):
        assert _clean_comment("Sort the list").endswith(".")

    def test_empty(self):
        assert _clean_comment("") == ""

    def test_separator_ignored(self):
        assert _clean_comment("# -------") == ""

    def test_short_ignored(self):
        assert _clean_comment("# ok") == ""

    def test_strips_hash(self):
        assert not _clean_comment("# Iterates through items").startswith("#")


# ── extract_block_pairs ────────────────────────────────────────────────────────

_SAMPLE = """\
def search_sorted(arr, target):
    low, high = 0, len(arr) - 1
    # Binary search: halve the search space each iteration
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
    return -1

def process_items(items):
    result = []
    # Iterate and filter valid items
    for item in items:
        if item > 0:
            result.append(item)
    return result
"""


class TestExtractBlockPairs:
    def test_finds_while_block(self):
        pairs = extract_block_pairs(_SAMPLE)
        assert any(p.block_type == "while" for p in pairs)

    def test_finds_for_block(self):
        pairs = extract_block_pairs(_SAMPLE)
        assert any(p.block_type == "for" for p in pairs)

    def test_comment_text_correct(self):
        pairs = extract_block_pairs(_SAMPLE)
        while_pairs = [p for p in pairs if p.block_type == "while"]
        assert any("Binary search" in p.target_text for p in while_pairs)

    def test_input_prefix(self):
        pairs = extract_block_pairs(_SAMPLE)
        for p in pairs:
            assert p.input_text.startswith("Comment Python block:")

    def test_no_pairs_from_plain_assignment(self):
        assert extract_block_pairs("# comment\nx = 1\n") == []

    def test_invalid_syntax_returns_empty(self):
        assert extract_block_pairs("def f(\n    broken }{") == []


# ── BlockCommenter ────────────────────────────────────────────────────────────

class _MockCodeT5:
    """Returns a fixed comment with confidence 0.75."""
    def generate(self, input_text, **kwargs):
        return "Iterates through the items.", 0.75


class _MockFF:
    name = "search_sorted"
    lineno = 1
    body_lines = 10


class _MockFC:
    cyclomatic_complexity = 4
    complexity_label = "moderate"


_COMPLEX_SRC = """\
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


class TestBlockCommenter:
    def test_complex_function_returns_list(self):
        from src.ml.block_commenter import BlockCommenter
        bc = BlockCommenter(_MockCodeT5())
        results = bc.generate(_COMPLEX_SRC, _MockFF(), _MockFC())
        assert isinstance(results, list)

    def test_comments_have_hash_prefix(self):
        from src.ml.block_commenter import BlockCommenter
        bc = BlockCommenter(_MockCodeT5())
        for lineno, col_offset, text in bc.generate(_COMPLEX_SRC, _MockFF(), _MockFC()):
            assert text.startswith("#")

    def test_linenos_are_ints(self):
        from src.ml.block_commenter import BlockCommenter
        bc = BlockCommenter(_MockCodeT5())
        for lineno, col_offset, text in bc.generate(_COMPLEX_SRC, _MockFF(), _MockFC()):
            assert isinstance(lineno, int)
            assert isinstance(col_offset, int)

    def test_simple_function_skipped(self):
        from src.ml.block_commenter import BlockCommenter

        class SimpleFF:
            name = "add"; lineno = 1; body_lines = 2

        class SimpleFC:
            cyclomatic_complexity = 1; complexity_label = "simple"

        bc = BlockCommenter(_MockCodeT5())
        assert bc.generate("def add(a, b):\n    return a + b\n", SimpleFF(), SimpleFC()) == []

    def test_low_confidence_filtered(self):
        from src.ml.block_commenter import BlockCommenter, MIN_CONFIDENCE

        class LowConf:
            def generate(self, *a, **k):
                return "text.", MIN_CONFIDENCE - 0.01

        bc = BlockCommenter(LowConf())
        assert bc.generate(_COMPLEX_SRC, _MockFF(), _MockFC()) == []


# ── save_block_corpus ─────────────────────────────────────────────────────────

class TestSaveBlockCorpus:
    def test_creates_json_and_csv(self):
        corpus = [
            BlockEntry("while", "Comment Python block: while x:", "Decrements x.", "f.py"),
            BlockEntry("for",   "Comment Python block: for i in a:", "Iterates array.", "f.py"),
        ]
        with tempfile.TemporaryDirectory() as d:
            info = save_block_corpus(corpus, d, base_name="test")
            assert os.path.exists(info["json_path"])
            assert os.path.exists(info["csv_path"])
            assert info["total_samples"] == 2
