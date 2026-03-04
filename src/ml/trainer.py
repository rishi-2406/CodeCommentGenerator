"""
Trainer — Week 9 ML
=====================
End-to-end training pipeline:
  1. Build dataset (seed corpus + optional extra files)
  2. Train/test split (80/20)
  3. Train TFIDFCommentModel
  4. Fit TemplateRankingModel
  5. Evaluate both on test split
  6. Save models + JSON reports to output_dir

Usage (from week9/ root):
    python -m src.main --train
    python -m src.main --train --output-dir outputs/model
"""
import json
import os
import pathlib
import random
from typing import List, Optional

from .dataset import build_dataset, Dataset
from .tfidf_model import TFIDFCommentModel
from .seq2seq_model import TemplateRankingModel
from .model_selector import ModelSelector
from .evaluator import evaluate_dataset, EvalReport


def _split_dataset(dataset: Dataset, test_ratio: float = 0.2, seed: int = 42):
    """Split Dataset into (train, test) Datasets."""
    from .dataset import Dataset as DS
    points = list(dataset.points)
    random.seed(seed)
    random.shuffle(points)
    split_idx = max(1, int(len(points) * (1 - test_ratio)))
    train_ds = DS(points=points[:split_idx])
    test_ds  = DS(points=points[split_idx:])
    return train_ds, test_ds


def train_and_evaluate(
    output_dir: str = "outputs",
    extra_files: Optional[List[str]] = None,
    test_ratio: float = 0.2,
    cv_folds: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Full training and evaluation pipeline.

    Args:
        output_dir:   Directory where models and reports are saved.
        extra_files:  Additional .py source files to augment the seed corpus.
        test_ratio:   Fraction of data reserved for evaluation.
        cv_folds:     Cross-validation folds for TFIDFCommentModel.
        verbose:      Print progress to stdout.

    Returns:
        dict with training_report and eval_report (JSON-serialisable).
    """
    model_dir = pathlib.Path(output_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str):
        if verbose:
            print(f"  [trainer] {msg}")

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    _log("Building dataset...")
    dataset = build_dataset(extra_source_files=extra_files)
    _log(f"  Dataset size: {len(dataset)} samples")

    if len(dataset) < 2:
        raise RuntimeError(
            "Dataset has fewer than 2 samples. Cannot train or evaluate."
        )

    train_ds, test_ds = _split_dataset(dataset, test_ratio=test_ratio)
    _log(f"  Train: {len(train_ds)}  |  Test: {len(test_ds)}")

    # ── 2. Train TFIDFCommentModel ────────────────────────────────────────────
    _log("Training TF-IDF + Logistic Regression model...")
    tfidf_model = TFIDFCommentModel()
    # Use reduced folds if dataset is small
    effective_folds = min(cv_folds, len(train_ds))
    tfidf_train_meta = tfidf_model.train(train_ds, cv_folds=effective_folds)
    _log(f"  CV accuracy: {tfidf_train_meta.get('cv_accuracy_mean', 'N/A')}")

    # ── 3. Fit TemplateRankingModel ───────────────────────────────────────────
    _log("Fitting template-ranking model...")
    template_model = TemplateRankingModel()
    template_model.fit()
    _log(f"  Template bank: {len(template_model._templates)} entries")

    # ── 4. Evaluate both models on test split ─────────────────────────────────
    _log("Evaluating on test split...")
    tfidf_eval  = evaluate_dataset(tfidf_model,    test_ds, model_name="TFIDFCommentModel")
    tmpl_eval   = evaluate_dataset(template_model, test_ds, model_name="TemplateRankingModel")

    # ── 5. Save models ────────────────────────────────────────────────────────
    tfidf_path = str(model_dir / "tfidf_model.pkl")
    tmpl_path  = str(model_dir / "template_model.pkl")
    tfidf_model.save(tfidf_path)
    template_model.save(tmpl_path)
    _log(f"  Models saved to {model_dir}/")

    # ── 6. Build reports ──────────────────────────────────────────────────────
    training_report = {
        "dataset_total": len(dataset),
        "train_size":    len(train_ds),
        "test_size":     len(test_ds),
        "tfidf_model":   tfidf_train_meta,
        "template_model": template_model.training_report(),
    }

    eval_report = {
        "tfidf_model":    tfidf_eval.to_dict(),
        "template_model": tmpl_eval.to_dict(),
        "summary": {
            "best_bleu4":     max(tfidf_eval.bleu4_mean, tmpl_eval.bleu4_mean),
            "best_rouge_l":   max(tfidf_eval.rouge_l_mean, tmpl_eval.rouge_l_mean),
            "best_exact_match": max(tfidf_eval.exact_match_rate, tmpl_eval.exact_match_rate),
        },
    }

    # ── 7. Save JSON reports ──────────────────────────────────────────────────
    reports_dir = pathlib.Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tr_path = reports_dir / "training_report.json"
    ev_path = reports_dir / "eval_report.json"

    with open(tr_path, "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)
    with open(ev_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    _log(f"  Training report : {tr_path}")
    _log(f"  Evaluation report: {ev_path}")

    return {"training_report": training_report, "eval_report": eval_report}


def load_model_selector(output_dir: str = "outputs",
                        min_confidence: float = 0.05) -> ModelSelector:
    """
    Load saved models and return a ModelSelector.

    Args:
        output_dir:    Directory containing the saved models.
        min_confidence: Minimum confidence threshold.

    Returns:
        ModelSelector configured with the loaded models.
    """
    model_dir = pathlib.Path(output_dir) / "model"
    tfidf_model    = None
    template_model = None

    tfidf_path = model_dir / "tfidf_model.pkl"
    tmpl_path  = model_dir / "template_model.pkl"

    if tfidf_path.exists():
        tfidf_model = TFIDFCommentModel.load(str(tfidf_path))

    if tmpl_path.exists():
        template_model = TemplateRankingModel.load(str(tmpl_path))

    if tfidf_model is None and template_model is None:
        # Cold start: train inline (slower but always works)
        dataset = build_dataset()
        tfidf_model = TFIDFCommentModel()
        tfidf_model.train(dataset, cv_folds=min(5, len(dataset)))
        template_model = TemplateRankingModel()
        template_model.fit()

    return ModelSelector(
        tfidf_model=tfidf_model,
        template_model=template_model,
        min_confidence=min_confidence,
    )
