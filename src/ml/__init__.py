"""
ML Package — AST-Feature NLP Comment Generation
================================================
Architecture:
  ast_feature_formatter  : FunctionFeature + FunctionContext → structured text
  ast_dataset_builder    : CodeSearchNet + stdlib → (AST text, docstring) pairs
  ast_comment_model      : T5-small fine-tuned on AST features → docstrings
  trainer                : end-to-end train + evaluate + save pipeline
  evaluator              : BLEU-4, ROUGE-L, exact-match metrics
"""
from .ast_feature_formatter import format_for_model
from .ast_dataset_builder   import (
    build_full_dataset, build_stdlib_dataset, save_dataset,
    load_dataset_from_json, ASTTrainPair,
)
from .evaluator import compute_bleu, compute_rouge, compute_exact_match, evaluate_dataset
from .trainer   import train_and_evaluate, load_ast_model

__all__ = [
    # Formatter
    "format_for_model",
    # Dataset
    "build_full_dataset", "build_stdlib_dataset", "save_dataset",
    "load_dataset_from_json", "ASTTrainPair",
    # Metrics
    "compute_bleu", "compute_rouge", "compute_exact_match", "evaluate_dataset",
    # Training
    "train_and_evaluate", "load_ast_model",
]
