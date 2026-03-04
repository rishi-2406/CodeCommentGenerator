"""
Model Selector — Week 9 ML
===========================
Combines TFIDFCommentModel and TemplateRankingModel.

Selection strategy:
  1. Run both models and retrieve their confidence scores.
  2. Return the prediction from the model with the higher confidence.
  3. If the chosen ML comment is very short (<10 chars) or empty, fall
     back to the rule-based generator result passed as `fallback`.
"""
from typing import Optional, Tuple


class ModelSelector:
    """
    Selects the best comment between two ML models, with an optional
    rule-based fallback.

    Args:
        tfidf_model:    Trained TFIDFCommentModel instance (or None).
        template_model: Fitted TemplateRankingModel instance (or None).
        min_confidence: Minimum confidence required to use ML output;
                        below this the fallback is used.
    """

    def __init__(self, tfidf_model=None, template_model=None,
                 min_confidence: float = 0.05):
        self._tfidf = tfidf_model
        self._template = template_model
        self._min_confidence = min_confidence

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        func_name: str,
        feature_vector,
        fallback: Optional[str] = None,
    ) -> Tuple[str, str, float]:
        """
        Generate the best comment for a function.

        Args:
            func_name:      Function identifier string.
            feature_vector: np.ndarray shape (FEATURE_DIM,).
            fallback:       Rule-based comment to use if confidence is low.

        Returns:
            (comment_text, source_model_name, confidence)
            source_model_name is one of: "tfidf", "template", "fallback"
        """
        candidates = []

        if self._tfidf is not None:
            try:
                text, conf = self._tfidf.predict(func_name, feature_vector)
                candidates.append(("tfidf", text, conf))
            except Exception:
                pass

        if self._template is not None:
            try:
                text, conf = self._template.predict(func_name, feature_vector)
                candidates.append(("template", text, conf))
            except Exception:
                pass

        if not candidates:
            fb = fallback or f'"""Handles {func_name}."""'
            return fb, "fallback", 0.0

        # Pick the candidate with the highest confidence
        best_name, best_text, best_conf = max(candidates, key=lambda x: x[2])

        # Use fallback if confidence is too low or text is trivially short
        if best_conf < self._min_confidence or len(best_text.strip()) < 10:
            fb = fallback or best_text
            return fb, "fallback", best_conf

        return best_text, best_name, best_conf

    def is_ready(self) -> bool:
        """Return True if at least one model is available."""
        return (self._tfidf is not None) or (self._template is not None)

    def summary(self) -> dict:
        """Return a summary of loaded models."""
        return {
            "tfidf_loaded": self._tfidf is not None,
            "template_loaded": self._template is not None,
            "min_confidence": self._min_confidence,
        }
