"""
Evaluator — Week 9 ML
======================
Computes standard NLP comment quality metrics:

  • BLEU-4  (sentence-level, NLTK implementation)
  • ROUGE-L (recall-oriented longest common subsequence)
  • Exact Match (normalised string equality)

All three are computed on (reference, hypothesis) docstring pairs and
averaged over a test dataset.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import re

# NLTK is used ONLY for BLEU; if unavailable we fall back to a simple
# unigram overlap so the module never hard-crashes.
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Strip quotes, docstring markers, leading/trailing spaces, lowercase."""
    text = text.strip()
    # Remove triple quotes and single-line docstring wrappers
    text = re.sub(r'^"""', '', text)
    text = re.sub(r'"""$', '', text)
    text = re.sub(r"^'''", '', text)
    text = re.sub(r"'''$", '', text)
    return text.strip().lower()


def _tokenise(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser."""
    return re.findall(r"[a-z0-9]+", _normalise(text))


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute Length of Longest Common Subsequence between two token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use compact DP to save memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


# ── Public metric functions ───────────────────────────────────────────────────

def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute sentence-level BLEU-4 score.

    Args:
        reference:  Ground-truth comment string.
        hypothesis: Model-generated comment string.

    Returns:
        BLEU-4 score in [0.0, 1.0].
    """
    ref_tokens = _tokenise(reference)
    hyp_tokens = _tokenise(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    if _NLTK_AVAILABLE:
        smoother = SmoothingFunction().method1
        try:
            return float(sentence_bleu(
                [ref_tokens], hyp_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoother,
            ))
        except Exception:
            pass

    # Fallback: simple unigram precision (BLEU-1 approximation)
    ref_set = set(ref_tokens)
    if not hyp_tokens:
        return 0.0
    matches = sum(1 for t in hyp_tokens if t in ref_set)
    return matches / len(hyp_tokens)


def compute_rouge(reference: str, hypothesis: str) -> float:
    """
    Compute ROUGE-L F1 score (LCS-based).

    Args:
        reference:  Ground-truth comment string.
        hypothesis: Model-generated comment string.

    Returns:
        ROUGE-L F1 score in [0.0, 1.0].
    """
    ref_tokens = _tokenise(reference)
    hyp_tokens = _tokenise(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall    = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(reference: str, hypothesis: str) -> float:
    """
    Compute exact match (normalised) as 0.0 or 1.0.

    Args:
        reference:  Ground-truth comment string.
        hypothesis: Model-generated comment string.

    Returns:
        1.0 if strings match after normalisation, else 0.0.
    """
    return 1.0 if _normalise(reference) == _normalise(hypothesis) else 0.0


# ── Aggregate evaluation ──────────────────────────────────────────────────────

@dataclass
class EvalReport:
    model_name: str = ""
    n_samples: int = 0
    bleu4_mean:      float = 0.0
    bleu4_std:       float = 0.0
    rouge_l_mean:    float = 0.0
    rouge_l_std:     float = 0.0
    exact_match_rate: float = 0.0
    per_function: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "n_samples": self.n_samples,
            "bleu4": {
                "mean": round(self.bleu4_mean, 4),
                "std":  round(self.bleu4_std, 4),
            },
            "rouge_l": {
                "mean": round(self.rouge_l_mean, 4),
                "std":  round(self.rouge_l_std, 4),
            },
            "exact_match_rate": round(self.exact_match_rate, 4),
            "per_function": self.per_function,
        }


def evaluate_dataset(model, dataset, model_name: str = "model") -> EvalReport:
    """
    Evaluate a model on a Dataset, computing BLEU-4, ROUGE-L, exact match.

    Args:
        model:      Any object with a .predict(func_name, feature_vector) method
                    that returns (comment_text, confidence).
        dataset:    Dataset instance (the test split).
        model_name: Label for the report.

    Returns:
        EvalReport with aggregated metrics.
    """
    import numpy as np

    bleu_scores  = []
    rouge_scores = []
    em_scores    = []
    per_fn       = []

    for point in dataset.points:
        try:
            pred_text, conf = model.predict(point.func_name, point.feature_vector)
        except Exception:
            pred_text, conf = "", 0.0

        b = compute_bleu(point.comment_text, pred_text)
        r = compute_rouge(point.comment_text, pred_text)
        e = compute_exact_match(point.comment_text, pred_text)

        bleu_scores.append(b)
        rouge_scores.append(r)
        em_scores.append(e)

        per_fn.append({
            "func_name":  point.func_name,
            "reference":  point.comment_text[:120],
            "hypothesis": pred_text[:120],
            "bleu4":      round(b, 4),
            "rouge_l":    round(r, 4),
            "exact_match": int(e),
            "confidence": round(conf, 4),
        })

    n = len(bleu_scores)
    return EvalReport(
        model_name=model_name,
        n_samples=n,
        bleu4_mean=float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        bleu4_std=float(np.std(bleu_scores)) if bleu_scores else 0.0,
        rouge_l_mean=float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        rouge_l_std=float(np.std(rouge_scores)) if rouge_scores else 0.0,
        exact_match_rate=float(np.mean(em_scores)) if em_scores else 0.0,
        per_function=per_fn,
    )
