"""
Trainer — Week 9 ML  (updated with CodeT5)
===========================================
End-to-end training pipeline:
  1. Build corpus from stdlib + installed packages + seed corpus
  2. Train/test split (80/20)
  3. Train TFIDFCommentModel
  4. Fit TemplateRankingModel
  5. Fine-tune CodeT5Model on the full corpus
  6. Evaluate all three on the test split
  7. Save models + JSON reports to output_dir

Usage (from week9/ root):
    python3 -m src.main --train
    python3 -m src.main --train --output-dir outputs/model
"""
import json
import os
import pathlib
import random
from typing import List, Optional

from .dataset import build_dataset, Dataset, DataPoint
from .tfidf_model import TFIDFCommentModel
from .seq2seq_model import TemplateRankingModel
from .model_selector import ModelSelector
from .evaluator import evaluate_dataset, EvalReport
from .corpus_builder import build_full_corpus, CorpusEntry, save_corpus


def _split_dataset(dataset: Dataset, test_ratio: float = 0.2, seed: int = 42):
    from .dataset import Dataset as DS
    points = list(dataset.points)
    random.seed(seed)
    random.shuffle(points)
    split_idx = max(1, int(len(points) * (1 - test_ratio)))
    return DS(points=points[:split_idx]), DS(points=points[split_idx:])


def _split_corpus(corpus: List[CorpusEntry], test_ratio: float = 0.2, seed: int = 42):
    data = list(corpus)
    random.seed(seed)
    random.shuffle(data)
    split_idx = max(1, int(len(data) * (1 - test_ratio)))
    return data[:split_idx], data[split_idx:]


def train_and_evaluate(
    output_dir: str = "outputs",
    extra_files: Optional[List[str]] = None,
    test_ratio: float = 0.2,
    cv_folds: int = 5,
    codet5_epochs: int = 3,
    codet5_batch_size: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Full training and evaluation pipeline including CodeT5 fine-tuning.

    Args:
        output_dir:        Where models + reports are saved.
        extra_files:       Additional .py files to augment the corpus.
        test_ratio:        Fraction of data for evaluation.
        cv_folds:          CV folds for TFIDFCommentModel.
        codet5_epochs:     Fine-tuning epochs for CodeT5 (3 recommended).
        codet5_batch_size: Batch size for CodeT5 (reduce to 4 if OOM).
        verbose:           Print progress.

    Returns:
        dict with training_report and eval_report.
    """
    model_dir = pathlib.Path(output_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg):
        if verbose:
            print(f"  [trainer] {msg}")

    # ── 1. Build dataset (for TF-IDF features model) ──────────────────────────
    _log("Building feature dataset (seed corpus) …")
    dataset = build_dataset(extra_source_files=extra_files)
    _log(f"  Feature dataset: {len(dataset)} samples")

    if len(dataset) < 2:
        raise RuntimeError("Dataset too small (<2 samples).")

    train_ds, test_ds = _split_dataset(dataset, test_ratio=test_ratio)

    # ── 2. Build large corpus (for CodeT5 fine-tuning) ────────────────────────
    _log("Building training corpus from stdlib + packages …")
    full_corpus = build_full_corpus(
        include_stdlib=True,
        include_packages=True,
        include_codesearchnet=True,
        verbose=verbose,
    )
    train_corpus, test_corpus = _split_corpus(full_corpus, test_ratio=test_ratio)
    _log(f"  Corpus — train: {len(train_corpus)}  test: {len(test_corpus)}")

    # Save corpus to disk for inspection
    corpus_info = save_corpus(full_corpus, output_dir=output_dir, base_name="training_corpus")
    _log(f"  Corpus saved: {corpus_info['total_samples']} samples → {corpus_info['json_path']}")

    # ── 3. Train TFIDFCommentModel ────────────────────────────────────────────
    _log("Training TF-IDF + Logistic Regression model …")
    tfidf_model = TFIDFCommentModel()
    effective_folds = min(cv_folds, len(train_ds))
    tfidf_train_meta = tfidf_model.train(train_ds, cv_folds=effective_folds)
    cv_acc = tfidf_train_meta.get("cv_accuracy_mean")
    _log(f"  TF-IDF CV accuracy: {cv_acc:.4f}" if cv_acc else "  TF-IDF CV accuracy: N/A")

    # ── 4. Fit TemplateRankingModel ───────────────────────────────────────────
    _log("Fitting template-ranking model …")
    template_model = TemplateRankingModel()
    template_model.fit()

    # ── 5. Fine-tune CodeT5 ───────────────────────────────────────────────────
    codet5_model = None
    codet5_train_meta = {"model": "CodeT5Model", "error": "not attempted"}
    try:
        from .codet5_model import CodeT5Model
        _log(f"Fine-tuning CodeT5 on {len(train_corpus)} samples …")
        codet5_model = CodeT5Model()
        codet5_train_meta = codet5_model.fine_tune(
            train_corpus,
            epochs=codet5_epochs,
            batch_size=codet5_batch_size,
            verbose=verbose,
        )
    except ImportError as e:
        _log(f"  [skip] CodeT5 unavailable: {e}")
        codet5_train_meta = {"model": "CodeT5Model", "error": str(e)}
    except Exception as e:
        _log(f"  [warn] CodeT5 training error: {e}")
        codet5_train_meta = {"model": "CodeT5Model", "error": str(e)}
        codet5_model = None

    # ── 6. Evaluate all models ────────────────────────────────────────────────
    _log("Evaluating on test split …")
    tfidf_eval  = evaluate_dataset(tfidf_model,    test_ds, model_name="TFIDFCommentModel")
    tmpl_eval   = evaluate_dataset(template_model, test_ds, model_name="TemplateRankingModel")

    codet5_eval_dict = None
    if codet5_model is not None and test_corpus:
        # Evaluate CodeT5 on the corpus test split using CorpusEntry objects
        from .evaluator import compute_bleu, compute_rouge, compute_exact_match, EvalReport
        import numpy as np
        bleu_scores, rouge_scores, em_scores, per_fn = [], [], [], []
        for entry in test_corpus[:50]:  # cap at 50 for speed
            try:
                pred, conf = codet5_model.generate(entry.input_text.replace("Summarize Python: ", ""))
                # Strip docstring quotes for metric computation
                pred_clean = pred.strip('"""').strip()
            except Exception:
                pred_clean, conf = "", 0.0
            b = compute_bleu(entry.target_text,  pred_clean)
            r = compute_rouge(entry.target_text, pred_clean)
            e = compute_exact_match(entry.target_text, pred_clean)
            bleu_scores.append(b); rouge_scores.append(r); em_scores.append(e)
            per_fn.append({
                "func_name":   entry.func_name,
                "reference":   entry.target_text[:120],
                "hypothesis":  pred_clean[:120],
                "bleu4":       round(b, 4),
                "rouge_l":     round(r, 4),
                "exact_match": int(e),
                "confidence":  round(conf, 4),
            })
        codet5_eval_report = EvalReport(
            model_name="CodeT5Model",
            n_samples=len(bleu_scores),
            bleu4_mean=float(np.mean(bleu_scores)) if bleu_scores else 0.0,
            bleu4_std=float(np.std(bleu_scores)) if bleu_scores else 0.0,
            rouge_l_mean=float(np.mean(rouge_scores)) if rouge_scores else 0.0,
            rouge_l_std=float(np.std(rouge_scores)) if rouge_scores else 0.0,
            exact_match_rate=float(np.mean(em_scores)) if em_scores else 0.0,
            per_function=per_fn,
        )
        codet5_eval_dict = codet5_eval_report.to_dict()
        _log(f"  CodeT5 BLEU-4={codet5_eval_report.bleu4_mean:.4f}  "
             f"ROUGE-L={codet5_eval_report.rouge_l_mean:.4f}")

    # ── 7. Save models ────────────────────────────────────────────────────────
    tfidf_model.save(str(model_dir / "tfidf_model.pkl"))
    template_model.save(str(model_dir / "template_model.pkl"))
    if codet5_model is not None:
        codet5_model.save(str(model_dir / "codet5"))
    _log(f"  Models saved to {model_dir}/")

    # ── 8. Build reports ──────────────────────────────────────────────────────
    training_report = {
        "dataset_total": len(dataset),
        "train_size":    len(train_ds),
        "test_size":     len(test_ds),
        "corpus_total":  len(full_corpus),
        "corpus_train":  len(train_corpus),
        "corpus_test":   len(test_corpus),
        "tfidf_model":   tfidf_train_meta,
        "template_model": template_model.training_report(),
        "codet5_model":  codet5_train_meta,
    }

    all_bleu  = [tfidf_eval.bleu4_mean, tmpl_eval.bleu4_mean]
    all_rouge = [tfidf_eval.rouge_l_mean, tmpl_eval.rouge_l_mean]
    all_em    = [tfidf_eval.exact_match_rate, tmpl_eval.exact_match_rate]
    if codet5_eval_dict:
        all_bleu.append(codet5_eval_dict["bleu4"]["mean"])
        all_rouge.append(codet5_eval_dict["rouge_l"]["mean"])
        all_em.append(codet5_eval_dict["exact_match_rate"])

    eval_report = {
        "tfidf_model":    tfidf_eval.to_dict(),
        "template_model": tmpl_eval.to_dict(),
        "codet5_model":   codet5_eval_dict,
        "summary": {
            "best_bleu4":       max(all_bleu),
            "best_rouge_l":     max(all_rouge),
            "best_exact_match": max(all_em),
            "corpus_size":      len(full_corpus),
        },
    }

    reports_dir = pathlib.Path(output_dir)
    with open(reports_dir / "training_report.json", "w") as f:
        json.dump(training_report, f, indent=2)
    with open(reports_dir / "eval_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)
    _log(f"  Reports saved to {output_dir}/")

    return {"training_report": training_report, "eval_report": eval_report}


def load_model_selector(output_dir: str = "outputs",
                        min_confidence: float = 0.05) -> ModelSelector:
    """
    Load saved models and return a ModelSelector (CodeT5 > TF-IDF > Template).
    If no saved models exist, trains inline.
    """
    model_dir = pathlib.Path(output_dir) / "model"
    tfidf_model    = None
    template_model = None
    codet5_model   = None

    tfidf_path    = model_dir / "tfidf_model.pkl"
    tmpl_path     = model_dir / "template_model.pkl"
    codet5_dir    = model_dir / "codet5"

    if tfidf_path.exists():
        tfidf_model = TFIDFCommentModel.load(str(tfidf_path))

    if tmpl_path.exists():
        template_model = TemplateRankingModel.load(str(tmpl_path))

    if codet5_dir.exists():
        try:
            from .codet5_model import CodeT5Model
            codet5_model = CodeT5Model.load(str(codet5_dir))
        except Exception as e:
            print(f"  [warn] Could not load CodeT5: {e}")

    if tfidf_model is None and template_model is None and codet5_model is None:
        # Cold start
        dataset = build_dataset()
        tfidf_model = TFIDFCommentModel()
        tfidf_model.train(dataset, cv_folds=min(5, len(dataset)))
        template_model = TemplateRankingModel()
        template_model.fit()

    return ModelSelector(
        tfidf_model=tfidf_model,
        template_model=template_model,
        codet5_model=codet5_model,
        min_confidence=min_confidence,
    )
