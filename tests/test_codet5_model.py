"""
Unit Tests — CodeT5 Model (Week 9)
====================================
Tests corpus_builder and CodeT5Model.
All tests that require transformers/torch are skipped gracefully
if those packages are not installed.
"""
import ast
import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Check transformers availability ──────────────────────────────────────────
try:
    import torch
    import transformers
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

requires_transformers = pytest.mark.skipif(
    not _HAS_TRANSFORMERS,
    reason="transformers + torch not installed"
)

from src.ml.corpus_builder import (
    build_stdlib_corpus, build_full_corpus,
    _clean_docstring, _build_signature, CorpusEntry,
)


# ── CorpusEntry / _clean_docstring tests ─────────────────────────────────────

class TestCleanDocstring:
    def test_simple_string(self):
        result = _clean_docstring("Returns the user record.")
        assert "Returns the user record" in result

    def test_adds_period(self):
        result = _clean_docstring("Returns the user record")
        assert result.endswith(".")

    def test_empty_returns_empty(self):
        assert _clean_docstring("") == ""

    def test_strips_rst_params(self):
        doc = "Compute the area.\n\n:param width: The width.\n:returns: float"
        result = _clean_docstring(doc)
        assert "Compute the area" in result
        assert len(result) < 40  # only first sentence

    def test_truncates_long_string(self):
        # Long string with multiple sentences: only first sentence kept
        sentences = ". ".join(["This is a sentence"] * 20) + "."
        result = _clean_docstring(sentences)
        # Must be shorter than the input
        assert len(result) < len(sentences)
        # Must still contain useful content
        assert len(result) > 5

    def test_first_paragraph_only(self):
        doc = "First sentence.\n\nSecond paragraph with more info."
        result = _clean_docstring(doc)
        assert "First sentence" in result
        assert "Second paragraph" not in result


class TestBuildSignature:
    def _parse_func(self, src: str) -> ast.FunctionDef:
        tree = ast.parse(src)
        return tree.body[0]

    def test_simple_signature(self):
        node = self._parse_func("def get_user(user_id: int) -> dict:\n    pass")
        sig = _build_signature(node)
        assert "get_user" in sig
        assert sig.endswith(":")

    def test_no_params(self):
        node = self._parse_func("def run():\n    pass")
        sig = _build_signature(node)
        assert "run()" in sig

    def test_return_annotation(self):
        node = self._parse_func("def area(w: float, h: float) -> float:\n    pass")
        sig = _build_signature(node)
        assert "float" in sig
        assert "->" in sig


# ── Corpus builder tests ──────────────────────────────────────────────────────

class TestBuildStdlibCorpus:
    def test_returns_nonempty_list(self):
        entries = build_stdlib_corpus(max_files=10)
        assert len(entries) > 0

    def test_entry_has_input_text(self):
        entries = build_stdlib_corpus(max_files=5)
        for e in entries:
            assert e.input_text.startswith("Summarize Python: def ")

    def test_entry_has_target_text(self):
        entries = build_stdlib_corpus(max_files=5)
        for e in entries:
            assert len(e.target_text) >= 6

    def test_no_private_functions(self):
        entries = build_stdlib_corpus(max_files=20)
        for e in entries:
            assert not e.func_name.startswith("_")


class TestBuildFullCorpus:
    def test_nonempty(self):
        corpus = build_full_corpus(
            include_stdlib=True, include_packages=False,
            max_stdlib_files=10, verbose=False,
        )
        assert len(corpus) > 0

    def test_deduplicated(self):
        corpus = build_full_corpus(
            include_stdlib=True, include_packages=False,
            max_stdlib_files=20, deduplicate=True, verbose=False,
        )
        keys = [(e.func_name, e.target_text[:50]) for e in corpus]
        assert len(keys) == len(set(keys))


# ── CodeT5Model tests ─────────────────────────────────────────────────────────

@requires_transformers
class TestCodeT5Model:
    """
    These tests fine-tune CodeT5 on 5 synthetic pairs (fast on CPU).
    They are skipped if transformers/torch is not installed.
    """

    def _make_corpus(self) -> list:
        from src.ml.corpus_builder import CorpusEntry
        return [
            CorpusEntry("get_user",       "Summarize Python: def get_user(uid: int) -> dict:", "Retrieves the user record."),
            CorpusEntry("set_value",       "Summarize Python: def set_value(key: str, val) -> None:", "Sets the value in the store."),
            CorpusEntry("calculate_area",  "Summarize Python: def calculate_area(w: float, h: float) -> float:", "Calculates the area."),
            CorpusEntry("validate_email",  "Summarize Python: def validate_email(email: str) -> bool:", "Validates an email address."),
            CorpusEntry("build_index",     "Summarize Python: def build_index(items: list) -> dict:", "Builds a lookup index."),
        ]

    def test_generate_before_finetune(self):
        from src.ml.codet5_model import CodeT5Model
        model = CodeT5Model()
        text, conf = model.generate("def get_user(uid: int) -> dict:")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_finetune_and_generate(self):
        from src.ml.codet5_model import CodeT5Model
        corpus = self._make_corpus()
        model = CodeT5Model()
        meta = model.fine_tune(corpus, epochs=1, batch_size=2, verbose=False)
        assert "final_train_loss" in meta
        assert meta["final_train_loss"] is not None
        text, conf = model.generate("def get_user(uid: int) -> dict:")
        assert isinstance(text, str) and len(text.strip()) > 0

    def test_confidence_in_range(self):
        from src.ml.codet5_model import CodeT5Model
        corpus = self._make_corpus()
        model = CodeT5Model()
        model.fine_tune(corpus, epochs=1, batch_size=2, verbose=False)
        _, conf = model.generate("def calculate_area(w: float, h: float) -> float:")
        assert 0.0 <= conf <= 1.0

    def test_save_and_load(self):
        from src.ml.codet5_model import CodeT5Model
        corpus = self._make_corpus()
        model = CodeT5Model()
        model.fine_tune(corpus, epochs=1, batch_size=2, verbose=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = CodeT5Model.load(tmpdir)
            text, conf = loaded.generate("def get_user(uid: int) -> dict:")
            assert isinstance(text, str)

    def test_predict_adapter(self):
        """ModelSelector API: predict(func_name, feature_vector)."""
        from src.ml.codet5_model import CodeT5Model
        model = CodeT5Model()
        text, conf = model.predict("get_user")
        assert isinstance(text, str)
