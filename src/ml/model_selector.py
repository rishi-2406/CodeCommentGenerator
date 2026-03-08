"""
Model Selector — Week 9 ML  (updated with CodeT5 priority)
============================================================
Priority order:
  1. CodeT5Model  (transformer-based, highest quality)
  2. TFIDFCommentModel  (fast, deterministic)
  3. TemplateRankingModel  (cosine-similarity fallback)
  4. Rule-based fallback  (always available)
"""
from typing import Optional, Tuple


class ModelSelector:
    """
    Selects the best comment across available ML models.

    Args:
        tfidf_model:    TFIDFCommentModel instance (or None).
        template_model: TemplateRankingModel instance (or None).
        codet5_model:   CodeT5Model instance (or None).
        min_confidence: Minimum confidence to prefer ML over fallback.
    """

    def __init__(self, tfidf_model=None, template_model=None,
                 codet5_model=None, min_confidence: float = 0.05):
        self._tfidf    = tfidf_model
        self._template = template_model
        self._codet5   = codet5_model
        self._min_confidence = min_confidence

    def predict(
        self,
        func_name: str,
        feature_vector,
        fallback: Optional[str] = None,
        func_signature: Optional[str] = None,
    ) -> Tuple[str, str, float]:
        """
        Generate the best comment for a function.

        Args:
            func_name:      Function identifier string.
            feature_vector: np.ndarray shape (FEATURE_DIM,).
            fallback:       Rule-based comment to use if confidence is low.
            func_signature: Full signature string for CodeT5 input (optional).

        Returns:
            (comment_text, source_model_name, confidence)
            source is one of: "codet5", "tfidf", "template", "fallback"
        """
        candidates = []

        # 1. CodeT5 — highest priority, needs a signature string
        if self._codet5 is not None:
            try:
                sig = func_signature or f"def {func_name}():"
                text, conf = self._codet5.generate(sig)
                candidates.append(("codet5", text, conf))
            except Exception:
                pass

        # 2. TF-IDF
        if self._tfidf is not None:
            try:
                text, conf = self._tfidf.predict(func_name, feature_vector)
                candidates.append(("tfidf", text, conf))
            except Exception:
                pass

        # 3. Template Ranking
        if self._template is not None:
            try:
                text, conf = self._template.predict(func_name, feature_vector)
                candidates.append(("template", text, conf))
            except Exception:
                pass

        if not candidates:
            fb = fallback or f'"""Handles {func_name}."""'
            return fb, "fallback", 0.0

        best_name, best_text, best_conf = max(candidates, key=lambda x: x[2])

        if best_conf < self._min_confidence or len(best_text.strip()) < 10:
            fb = fallback or best_text
            return fb, "fallback", best_conf

        return best_text, best_name, best_conf

    def is_ready(self) -> bool:
        return any(m is not None for m in [self._codet5, self._tfidf, self._template])

    @property
    def codet5_model(self):
        """Return the CodeT5Model instance (or None) for direct use."""
        return self._codet5

    def summary(self) -> dict:
        return {
            "codet5_loaded":   self._codet5   is not None,
            "tfidf_loaded":    self._tfidf    is not None,
            "template_loaded": self._template is not None,
            "min_confidence":  self._min_confidence,
        }
