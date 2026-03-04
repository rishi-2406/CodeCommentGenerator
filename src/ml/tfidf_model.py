"""
TF-IDF + Logistic Regression Comment Model — Week 9 ML
=======================================================
Learns to predict a comment *label bucket* from function feature vectors,
then selects the best pre-generated comment template for that bucket.

Architecture:
  - Input : raw function name string (fed through TfidfVectorizer)
             concatenated with normalised numeric feature vector
  - Model : LogisticRegression (one-vs-rest, lbfgs)
  - Output: predicted verb-bucket label  → used to select a template

Why TF-IDF + LR?
  - Fully offline (sklearn only)
  - Interpretable (coefficients per feature × token)
  - Fast training on small corpora
  - Deterministic and reproducible
"""
import pickle
import pathlib
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from .feature_vectors import FEATURE_DIM, VERB_BUCKET_NAMES
from .dataset import Dataset


# ── Comment templates per verb bucket ────────────────────────────────────────
# Each bucket maps to a list of templates; {name} is replaced with the
# natural-language form of the function name.

_BUCKET_TEMPLATES = {
    "get_fetch":    [
        '"""Retrieves {name}."""',
        '"""Fetches and returns {name}."""',
        '"""Loads {name} from the data source."""',
    ],
    "set_update":   [
        '"""Sets {name}."""',
        '"""Updates the {name} value."""',
        '"""Writes {name} to the backing store."""',
    ],
    "add_insert":   [
        '"""Adds {name} to the collection."""',
        '"""Inserts {name} into the data structure."""',
        '"""Appends {name} to the sequence."""',
    ],
    "remove":       [
        '"""Removes {name} from the collection."""',
        '"""Deletes the specified {name}."""',
        '"""Clears {name} from the store."""',
    ],
    "compute":      [
        '"""Computes {name} and returns the result."""',
        '"""Calculates {name}."""',
        '"""Returns the computed value of {name}."""',
    ],
    "find_search":  [
        '"""Searches for {name} and returns the match."""',
        '"""Finds {name} within the dataset."""',
        '"""Looks up {name} and returns the result."""',
    ],
    "check_valid":  [
        '"""Checks whether {name} satisfies the constraint."""',
        '"""Validates {name} and returns a boolean."""',
        '"""Returns True if {name} is valid."""',
    ],
    "parse_format": [
        '"""Parses {name} and returns structured data."""',
        '"""Formats {name} into the target representation."""',
        '"""Converts {name} between formats."""',
    ],
    "process_run":  [
        '"""Processes {name} through the pipeline."""',
        '"""Executes {name} and handles the result."""',
        '"""Runs the {name} routine."""',
    ],
    "build_create": [
        '"""Builds and returns a new {name}."""',
        '"""Creates {name} with the given parameters."""',
        '"""Initializes and returns {name}."""',
    ],
    "send_receive": [
        '"""Sends {name} to the target endpoint."""',
        '"""Transmits {name} and waits for acknowledgement."""',
        '"""Logs and dispatches {name}."""',
    ],
    "test_assert":  [
        '"""Tests {name} for correctness."""',
        '"""Asserts that {name} behaves as expected."""',
        '"""Validates {name} via automated checks."""',
    ],
    "__default__":  [
        '"""Handles {name}."""',
        '"""Performs the {name} operation."""',
        '"""Executes {name}."""',
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _humanise(func_name: str) -> str:
    """Convert snake_case/CamelCase function name to plain English."""
    import re
    parts = func_name.split("_")
    tokens = []
    for part in parts:
        sub = re.sub(r'([A-Z][a-z]+)', r' \1', part)
        sub = re.sub(r'([A-Z]+)([A-Z][a-z])', r' \1 \2', sub)
        tokens.extend(sub.strip().lower().split())
    return " ".join(t for t in tokens if t)


def _template_for_bucket(bucket: str, func_name: str, variant: int = 0) -> str:
    """Select a template for the given bucket and fill in the function name."""
    templates = _BUCKET_TEMPLATES.get(bucket, _BUCKET_TEMPLATES["__default__"])
    tmpl = templates[variant % len(templates)]
    return tmpl.format(name=_humanise(func_name))


# ── Model class ───────────────────────────────────────────────────────────────

class TFIDFCommentModel:
    """
    TF-IDF + Logistic Regression comment generation model.

    Trained to predict the verb-bucket label from function name tokens
    and numeric AST features. The predicted bucket then selects the
    best comment template.
    """

    def __init__(self, max_iter: int = 500, C: float = 1.0):
        self._tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            max_features=512,
        )
        self._lr = LogisticRegression(
            max_iter=max_iter, C=C, solver="lbfgs",
            class_weight="balanced",
        )
        self._label_encoder = LabelEncoder()
        self._trained = False
        self._cv_scores: List[float] = []
        self._train_size: int = 0

    # ── training ─────────────────────────────────────────────────────────────

    def _build_input(self, func_names: List[str], X_num: np.ndarray) -> np.ndarray:
        """Combine TF-IDF text features with numeric features."""
        tfidf_mat = self._tfidf.transform(func_names).toarray()
        # Normalise numeric part
        mx = np.abs(X_num).max(axis=0, keepdims=True) + 1e-8
        X_norm = X_num / mx
        return np.concatenate([tfidf_mat, X_norm], axis=1)

    def _infer_bucket_labels(self, func_names: List[str]) -> List[str]:
        """Derive verb-bucket label from function name (weak supervision)."""
        from .feature_vectors import _split_name, _VERB_BUCKET_INDEX, VERB_BUCKET_NAMES
        labels = []
        for name in func_names:
            label = "__default__"
            for tok in _split_name(name):
                if tok in _VERB_BUCKET_INDEX:
                    label = VERB_BUCKET_NAMES[_VERB_BUCKET_INDEX[tok]]
                    break
            labels.append(label)
        return labels

    def train(self, dataset: Dataset, cv_folds: int = 5) -> dict:
        """
        Train the model on the given Dataset.

        Args:
            dataset: Dataset with at least 5 data points.
            cv_folds: Number of cross-validation folds.

        Returns:
            dict with training metadata.
        """
        func_names = [p.func_name for p in dataset.points]
        X_num = dataset.X()
        labels = self._infer_bucket_labels(func_names)

        # Fit TF-IDF on function names
        self._tfidf.fit(func_names)
        X_combined = self._build_input(func_names, X_num)

        # Encode labels
        y_enc = self._label_encoder.fit_transform(labels)

        # Cross-validate (only if enough samples per class)
        cv_folds = min(cv_folds, len(dataset))
        if cv_folds >= 2:
            import numpy as _np
            unique, counts = _np.unique(y_enc, return_counts=True)
            min_class_count = int(counts.min())
            cv_folds = min(cv_folds, min_class_count)

        if cv_folds >= 2:
            self._cv_scores = cross_val_score(
                self._lr, X_combined, y_enc,
                cv=cv_folds, scoring="accuracy"
            ).tolist()
        else:
            self._cv_scores = []

        # Final fit on all data
        self._lr.fit(X_combined, y_enc)
        self._trained = True
        self._train_size = len(dataset)

        return {
            "train_size": self._train_size,
            "cv_folds": cv_folds,
            "cv_accuracy_mean": float(np.mean(self._cv_scores)) if self._cv_scores else None,
            "cv_accuracy_std":  float(np.std(self._cv_scores))  if self._cv_scores else None,
            "n_classes": len(self._label_encoder.classes_),
            "classes": self._label_encoder.classes_.tolist(),
        }

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, func_name: str, feature_vector: np.ndarray, variant: int = 0) -> Tuple[str, float]:
        """
        Predict a comment for the given function.

        Args:
            func_name: The function identifier (e.g. "calculate_area")
            feature_vector: np.ndarray of shape (FEATURE_DIM,)
            variant: Template variant index (0 = first/default)

        Returns:
            (comment_text, confidence_score)
        """
        if not self._trained:
            return _template_for_bucket("__default__", func_name, variant), 0.0

        X_num = feature_vector.reshape(1, -1)
        X_combined = self._build_input([func_name], X_num)
        proba = self._lr.predict_proba(X_combined)[0]
        best_idx = int(np.argmax(proba))
        confidence = float(proba[best_idx])
        bucket = self._label_encoder.inverse_transform([best_idx])[0]
        comment = _template_for_bucket(bucket, func_name, variant)
        return comment, confidence

    def training_report(self) -> dict:
        """Return a JSON-serialisable training report."""
        return {
            "model": "TFIDFCommentModel",
            "trained": self._trained,
            "train_size": self._train_size,
            "cv_accuracy_mean": float(np.mean(self._cv_scores)) if self._cv_scores else None,
            "cv_accuracy_std":  float(np.std(self._cv_scores)) if self._cv_scores else None,
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize model to disk."""
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TFIDFCommentModel":
        """Load a previously saved model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
