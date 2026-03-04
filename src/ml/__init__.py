"""
ML/AI package — Week 9 (updated with CodeT5)
=============================================
Sub-modules:
  corpus_builder   : builds (signature, docstring) corpus from stdlib + packages
  feature_vectors  : extract numeric vectors from AST features
  dataset          : build labelled (X, y) dataset for TF-IDF model
  tfidf_model      : TF-IDF + LogisticRegression comment classifier
  seq2seq_model    : lightweight offline template-ranking model (fallback)
  codet5_model     : CodeT5 fine-tuned seq2seq model (primary)
  model_selector   : pick best model prediction (CodeT5 > TF-IDF > template)
  evaluator        : BLEU-4, ROUGE-L, exact-match metrics
  trainer          : end-to-end train + evaluate + save
"""
from .feature_vectors import extract_feature_vector, FEATURE_NAMES
from .dataset import build_dataset, Dataset, DataPoint
from .tfidf_model import TFIDFCommentModel
from .seq2seq_model import TemplateRankingModel
from .model_selector import ModelSelector
from .evaluator import compute_bleu, compute_rouge, compute_exact_match, evaluate_dataset
from .trainer import train_and_evaluate

__all__ = [
    "extract_feature_vector", "FEATURE_NAMES",
    "build_dataset", "Dataset", "DataPoint",
    "TFIDFCommentModel",
    "TemplateRankingModel",
    "ModelSelector",
    "compute_bleu", "compute_rouge", "compute_exact_match", "evaluate_dataset",
    "train_and_evaluate",
]
