"""
Unit Tests — ML Feature Vectors (Week 9)
=========================================
Tests that extract_feature_vector returns correct shapes, types, and
meaningful content for various FunctionFeature + FunctionContext inputs.
"""
import sys
import os
import numpy as np
import pytest

# Allow running from the week9/ root:  python -m pytest tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ml.feature_vectors import (
    extract_feature_vector, FEATURE_DIM, FEATURE_NAMES,
    _split_name, _verb_one_hot, _complexity_one_hot,
)
from src.ast_extractor import FunctionFeature, ParamFeature


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ff(name="get_user", params=None, body_lines=5, loops=0,
             conditionals=0, calls=None, is_method=False, is_async=False,
             return_annotation=None, has_docstring=False):
    params = params or [ParamFeature(name="user_id", annotation="int")]
    calls = calls or []
    return FunctionFeature(
        node_id=f"func_{name}_1",
        name=name,
        lineno=1,
        col_offset=0,
        params=params,
        return_annotation=return_annotation,
        decorators=[],
        has_docstring=has_docstring,
        docstring=None,
        body_lines=body_lines,
        calls_made=calls,
        loops=loops,
        conditionals=conditionals,
        is_method=is_method,
        parent_class=None,
        is_async=is_async,
    )


class _MockFC:
    def __init__(self, cc=1, label="simple"):
        self.cyclomatic_complexity = cc
        self.complexity_label = label


# ── Test cases ────────────────────────────────────────────────────────────────

class TestSplitName:
    def test_snake_case(self):
        assert _split_name("get_user") == ["get", "user"]

    def test_camel_case(self):
        tokens = _split_name("getUserById")
        assert "get" in tokens
        assert "user" in tokens

    def test_single_word(self):
        assert _split_name("process") == ["process"]

    def test_empty(self):
        assert _split_name("") == []


class TestVerbOneHot:
    def test_get_bucket(self):
        vec = _verb_one_hot("get_user")
        assert vec.sum() == 1.0  # exactly one bucket active

    def test_unknown_name(self):
        vec = _verb_one_hot("xyz_abc")
        assert vec.sum() == 0.0  # no bucket matched

    def test_compute_bucket(self):
        vec = _verb_one_hot("calculate_area")
        assert vec.sum() == 1.0


class TestComplexityOneHot:
    def test_simple(self):
        vec = _complexity_one_hot("simple")
        assert vec[0] == 1.0 and vec.sum() == 1.0

    def test_very_complex(self):
        vec = _complexity_one_hot("very_complex")
        assert vec[3] == 1.0

    def test_unknown_falls_to_simple(self):
        vec = _complexity_one_hot("nonexistent_label")
        assert vec[0] == 1.0


class TestExtractFeatureVector:
    def test_output_shape(self):
        ff = _make_ff()
        vec = extract_feature_vector(ff)
        assert vec.shape == (FEATURE_DIM,)

    def test_output_dtype(self):
        ff = _make_ff()
        vec = extract_feature_vector(ff)
        assert vec.dtype == np.float32

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_n_params(self):
        ff = _make_ff(params=[ParamFeature("a"), ParamFeature("b")])
        vec = extract_feature_vector(ff)
        assert vec[0] == 2.0  # feature index 0 = n_params

    def test_self_skipped_in_params(self):
        ff = _make_ff(params=[ParamFeature("self"), ParamFeature("x")])
        vec = extract_feature_vector(ff)
        assert vec[0] == 1.0  # only "x" counted

    def test_body_lines(self):
        ff = _make_ff(body_lines=42)
        vec = extract_feature_vector(ff)
        assert vec[1] == 42.0

    def test_is_method_flag(self):
        ff = _make_ff(is_method=True)
        vec = extract_feature_vector(ff)
        assert vec[6] == 1.0

    def test_is_async_flag(self):
        ff = _make_ff(is_async=True)
        vec = extract_feature_vector(ff)
        assert vec[7] == 1.0

    def test_has_return_annotation(self):
        ff = _make_ff(return_annotation="int")
        vec = extract_feature_vector(ff)
        assert vec[8] == 1.0

    def test_no_return_annotation(self):
        ff = _make_ff(return_annotation=None)
        vec = extract_feature_vector(ff)
        assert vec[8] == 0.0

    def test_with_fc(self):
        ff = _make_ff()
        fc = _MockFC(cc=5, label="moderate")
        vec = extract_feature_vector(ff, fc)
        assert vec[4] == 5.0   # cyclomatic_complexity
        assert vec[11] == 1.0  # complexity_moderate one-hot

    def test_no_negative_values(self):
        ff = _make_ff(body_lines=0, loops=0, conditionals=0)
        vec = extract_feature_vector(ff)
        assert (vec >= 0).all()

    def test_loops_conditionals(self):
        ff = _make_ff(loops=3, conditionals=5)
        vec = extract_feature_vector(ff)
        assert vec[2] == 3.0
        assert vec[3] == 5.0
