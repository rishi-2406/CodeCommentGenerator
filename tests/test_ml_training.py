"""
Unit Tests — ML Training Smoke Test (Week 9)
=============================================
Verifies that:
  • The dataset builder produces DataPoints with correct shapes.
  • TFIDFCommentModel can train on a tiny in-memory dataset.
  • TemplateRankingModel can fit and predict without errors.
  • ModelSelector selects a result from at least one model.
  • trainer.train_and_evaluate() saves both report JSON files.
"""
import sys
import os
import json
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ml.feature_vectors import FEATURE_DIM
from src.ml.dataset import build_dataset, Dataset, DataPoint
from src.ml.tfidf_model import TFIDFCommentModel
from src.ml.seq2seq_model import TemplateRankingModel
from src.ml.model_selector import ModelSelector
from src.ml.trainer import train_and_evaluate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_dataset(n: int = 10) -> Dataset:
    """Generate a minimal synthetic Dataset for smoke tests."""
    rng = np.random.default_rng(42)
    names    = ["get_user", "set_value", "calculate_area", "parse_config",
                "validate_email", "find_duplicates", "build_index",
                "send_notification", "run_pipeline", "generate_report"]
    comments = [
        '"""Retrieves the user record."""',
        '"""Sets the value."""',
        '"""Calculates the area."""',
        '"""Parses the config file."""',
        '"""Validates an email address."""',
        '"""Finds duplicate items."""',
        '"""Builds a lookup index."""',
        '"""Sends a notification."""',
        '"""Runs the pipeline."""',
        '"""Generates a report."""',
    ]
    points = []
    for i in range(min(n, len(names))):
        vec = rng.random(FEATURE_DIM).astype(np.float32)
        points.append(DataPoint(
            func_name=names[i],
            comment_text=comments[i],
            feature_vector=vec,
        ))
    return Dataset(points=points)


# ── Dataset tests ─────────────────────────────────────────────────────────────

class TestBuildDataset:
    def test_seed_corpus_nonempty(self):
        ds = build_dataset()
        assert len(ds) > 0

    def test_feature_vector_shape(self):
        ds = build_dataset()
        for point in ds.points:
            assert point.feature_vector.shape == (FEATURE_DIM,)

    def test_feature_vector_dtype(self):
        ds = build_dataset()
        for point in ds.points:
            assert point.feature_vector.dtype == np.float32

    def test_comment_text_nonempty(self):
        ds = build_dataset()
        for point in ds.points:
            assert len(point.comment_text.strip()) > 0

    def test_X_matrix_shape(self):
        ds = build_dataset()
        X = ds.X()
        assert X.shape == (len(ds), FEATURE_DIM)

    def test_y_list_length(self):
        ds = build_dataset()
        assert len(ds.y()) == len(ds)


# ── TFIDFCommentModel tests ───────────────────────────────────────────────────

class TestTFIDFCommentModel:
    def test_train_returns_metadata(self):
        ds = _tiny_dataset()
        model = TFIDFCommentModel()
        meta = model.train(ds, cv_folds=2)
        assert "train_size" in meta
        assert meta["train_size"] == len(ds)

    def test_predict_returns_string(self):
        ds = _tiny_dataset()
        model = TFIDFCommentModel()
        model.train(ds, cv_folds=2)
        text, conf = model.predict("get_user", ds.points[0].feature_vector)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_predict_confidence_in_range(self):
        ds = _tiny_dataset()
        model = TFIDFCommentModel()
        model.train(ds, cv_folds=2)
        _, conf = model.predict("get_user", ds.points[0].feature_vector)
        assert 0.0 <= conf <= 1.0

    def test_untrained_predict_still_works(self):
        model = TFIDFCommentModel()
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        text, conf = model.predict("my_func", vec)
        assert isinstance(text, str)

    def test_save_and_load(self):
        ds = _tiny_dataset()
        model = TFIDFCommentModel()
        model.train(ds, cv_folds=2)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            model.save(tmp_path)
            loaded = TFIDFCommentModel.load(tmp_path)
            text, conf = loaded.predict("get_user", ds.points[0].feature_vector)
            assert isinstance(text, str)
        finally:
            os.unlink(tmp_path)

    def test_training_report(self):
        ds = _tiny_dataset()
        model = TFIDFCommentModel()
        model.train(ds, cv_folds=2)
        report = model.training_report()
        assert report["trained"] is True
        assert report["train_size"] > 0


# ── TemplateRankingModel tests ────────────────────────────────────────────────

class TestTemplateRankingModel:
    def test_fit_and_predict(self):
        model = TemplateRankingModel()
        model.fit()
        text, conf = model.predict("get_user")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_confidence_in_range(self):
        model = TemplateRankingModel()
        model.fit()
        _, conf = model.predict("calculate_area")
        assert 0.0 <= conf <= 1.0

    def test_auto_fit_on_first_predict(self):
        model = TemplateRankingModel()
        text, conf = model.predict("validate_email")
        assert isinstance(text, str)

    def test_template_bank_nonempty(self):
        model = TemplateRankingModel()
        assert len(model._templates) > 0

    def test_save_and_load(self):
        model = TemplateRankingModel()
        model.fit()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            model.save(tmp_path)
            loaded = TemplateRankingModel.load(tmp_path)
            text, conf = loaded.predict("get_user")
            assert isinstance(text, str)
        finally:
            os.unlink(tmp_path)


# ── ModelSelector tests ───────────────────────────────────────────────────────

class TestModelSelector:
    def _make_selector(self):
        ds = _tiny_dataset()
        tfidf = TFIDFCommentModel()
        tfidf.train(ds, cv_folds=2)
        tmpl = TemplateRankingModel()
        tmpl.fit()
        return ModelSelector(tfidf_model=tfidf, template_model=tmpl)

    def test_is_ready(self):
        sel = self._make_selector()
        assert sel.is_ready() is True

    def test_not_ready_when_empty(self):
        sel = ModelSelector()
        assert sel.is_ready() is False

    def test_predict_returns_tuple(self):
        sel = self._make_selector()
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        text, source, conf = sel.predict("get_user", vec)
        assert isinstance(text, str)
        assert source in ("tfidf", "template", "fallback")
        assert 0.0 <= conf <= 1.0

    def test_fallback_when_no_models(self):
        sel = ModelSelector()
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        text, source, conf = sel.predict("my_func", vec, fallback="# fallback")
        assert source == "fallback"

    def test_summary(self):
        sel = self._make_selector()
        s = sel.summary()
        assert s["tfidf_loaded"] is True
        assert s["template_loaded"] is True


# ── train_and_evaluate smoke test ─────────────────────────────────────────────

class TestTrainAndEvaluate:
    def test_produces_json_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_and_evaluate(output_dir=tmpdir, verbose=False)
            assert "training_report" in result
            assert "eval_report"     in result

            tr_path = os.path.join(tmpdir, "training_report.json")
            ev_path = os.path.join(tmpdir, "eval_report.json")
            assert os.path.exists(tr_path)
            assert os.path.exists(ev_path)

            with open(tr_path) as f:
                tr = json.load(f)
            with open(ev_path) as f:
                ev = json.load(f)

            assert tr["dataset_total"] > 0
            assert "tfidf_model"    in ev
            assert "template_model" in ev

    def test_saves_model_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_and_evaluate(output_dir=tmpdir, verbose=False)
            model_dir = os.path.join(tmpdir, "model")
            assert os.path.exists(os.path.join(model_dir, "tfidf_model.pkl"))
            assert os.path.exists(os.path.join(model_dir, "template_model.pkl"))
