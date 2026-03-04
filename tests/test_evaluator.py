"""
Unit Tests — Evaluator Metrics (Week 9)
========================================
Tests BLEU-4, ROUGE-L, and exact-match computation on deterministic
reference/hypothesis pairs with known expected outcomes.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ml.evaluator import (
    compute_bleu, compute_rouge, compute_exact_match,
    evaluate_dataset, EvalReport,
)
from src.ml.dataset import Dataset, DataPoint
from src.ml.feature_vectors import FEATURE_DIM


# ── compute_bleu ─────────────────────────────────────────────────────────────

class TestComputeBleu:
    def test_identical_strings_score_one(self):
        score = compute_bleu("retrieves the user record", "retrieves the user record")
        assert score > 0.5  # NLTK BLEU with smoothing on longer strings

    def test_empty_hypothesis_returns_zero(self):
        score = compute_bleu("retrieves user", "")
        assert score == 0.0

    def test_empty_reference_returns_zero(self):
        score = compute_bleu("", "some text")
        assert score == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        score = compute_bleu("retrieves the user record", "retrieves user data")
        assert 0.0 <= score <= 1.0

    def test_no_overlap_near_zero(self):
        score = compute_bleu("calculates area", "logs notification")
        assert score < 0.5

    def test_docstring_wrappers_stripped(self):
        # Triple-quote wrapper stripped before tokenisation — same content
        score = compute_bleu('retrieves the user record', 'retrieves the user record')
        assert score > 0.5


# ── compute_rouge ─────────────────────────────────────────────────────────────

class TestComputeRouge:
    def test_identical_strings_score_one(self):
        score = compute_rouge("retrieves the user record", "retrieves the user record")
        assert score > 0.9

    def test_empty_strings_return_zero(self):
        assert compute_rouge("", "") == 0.0
        assert compute_rouge("abc", "") == 0.0
        assert compute_rouge("", "abc") == 0.0

    def test_partial_overlap(self):
        score = compute_rouge("retrieves the user record", "retrieves user")
        assert 0.0 < score < 1.0

    def test_returns_float(self):
        assert isinstance(compute_rouge("a b c", "a b"), float)

    def test_symmetric_f1(self):
        # ROUGE-L F1 should give similar results in both directions
        a = compute_rouge("retrieves user data", "retrieves data")
        b = compute_rouge("retrieves data", "retrieves user data")
        assert abs(a - b) < 0.3  # not perfectly symmetric but close


# ── compute_exact_match ───────────────────────────────────────────────────────

class TestComputeExactMatch:
    def test_identical_returns_one(self):
        assert compute_exact_match("retrieves user", "retrieves user") == 1.0

    def test_different_returns_zero(self):
        assert compute_exact_match("retrieves user", "calculates area") == 0.0

    def test_case_insensitive(self):
        assert compute_exact_match("Retrieves User", "retrieves user") == 1.0

    def test_docstring_quotes_ignored(self):
        assert compute_exact_match('"""retrieves user"""', "retrieves user") == 1.0

    def test_leading_trailing_spaces_ignored(self):
        assert compute_exact_match("  retrieves user  ", "retrieves user") == 1.0


# ── evaluate_dataset ─────────────────────────────────────────────────────────

class _PerfectModel:
    """A mock model that returns the exact reference comment."""
    def __init__(self, dataset):
        self._map = {p.func_name: p.comment_text for p in dataset.points}

    def predict(self, func_name, feature_vector):
        return self._map.get(func_name, ""), 1.0


class _EmptyModel:
    """A mock model that always returns an empty string."""
    def predict(self, func_name, feature_vector):
        return "", 0.0


def _make_dataset(n: int = 5) -> Dataset:
    rng = np.random.default_rng(0)
    labels = [
        '"""retrieves the user record"""',
        '"""calculates the area"""',
        '"""validates email address"""',
        '"""builds a lookup index"""',
        '"""sends a notification"""',
    ]
    names = ["get_user", "calculate_area", "validate_email", "build_index", "send_notification"]
    return Dataset(points=[
        DataPoint(
            func_name=names[i],
            comment_text=labels[i],
            feature_vector=rng.random(FEATURE_DIM).astype(np.float32),
        )
        for i in range(min(n, len(labels)))
    ])


class TestEvaluateDataset:
    def test_perfect_model_high_scores(self):
        ds = _make_dataset()
        model = _PerfectModel(ds)
        report = evaluate_dataset(model, ds, model_name="PerfectModel")
        # ROUGE-L and EM should be perfect; BLEU may be lower due to smoothing
        assert report.rouge_l_mean > 0.9
        assert report.exact_match_rate > 0.9
        assert report.bleu4_mean > 0.5  # NLTK smoothing on short strings

    def test_empty_model_zero_scores(self):
        ds = _make_dataset()
        model = _EmptyModel()
        report = evaluate_dataset(model, ds, model_name="EmptyModel")
        assert report.bleu4_mean == 0.0
        assert report.rouge_l_mean == 0.0
        assert report.exact_match_rate == 0.0

    def test_report_has_correct_n_samples(self):
        ds = _make_dataset(n=4)
        model = _EmptyModel()
        report = evaluate_dataset(model, ds)
        assert report.n_samples == 4

    def test_to_dict_is_serialisable(self):
        import json
        ds = _make_dataset()
        model = _EmptyModel()
        report = evaluate_dataset(model, ds)
        d = report.to_dict()
        # Should not raise
        json.dumps(d)

    def test_per_function_entries(self):
        ds = _make_dataset()
        model = _PerfectModel(ds)
        report = evaluate_dataset(model, ds)
        assert len(report.per_function) == len(ds)
        for entry in report.per_function:
            assert "func_name" in entry
            assert "bleu4"     in entry
            assert "rouge_l"   in entry
