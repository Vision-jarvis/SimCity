"""
Toxicity classification using Detoxify (Unitary AI).
Detects: toxicity, severe toxicity, obscenity, threat, insult, identity attack.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ToxicityClassifier:
    """
    Multi-label toxicity classifier using Detoxify.
    Returns scores for multiple toxicity dimensions.
    """

    def __init__(self, threshold: Optional[float] = None, device: Optional[str] = None):
        from nlp.config import TOXICITY_THRESHOLD, DEVICE

        self.threshold = threshold or TOXICITY_THRESHOLD
        self.device = device or DEVICE
        self._model = None

    @property
    def model(self):
        """Lazy-load Detoxify model."""
        if self._model is None:
            try:
                from detoxify import Detoxify

                logger.info("Loading Detoxify model...")
                self._model = Detoxify(
                    "original",
                    device=self.device,
                )
                logger.info("Detoxify model loaded.")
            except ImportError:
                raise ImportError(
                    "detoxify is required. Install with: pip install detoxify"
                )
        return self._model

    def classify(self, text: str) -> Dict[str, float]:
        """
        Classify a single text across toxicity dimensions.

        Returns:
            Dict mapping dimension names to scores in [0, 1]:
            {toxicity, severe_toxicity, obscene, threat, insult, identity_attack}
        """
        if not text or not text.strip():
            return {
                "toxicity": 0.0,
                "severe_toxicity": 0.0,
                "obscene": 0.0,
                "threat": 0.0,
                "insult": 0.0,
                "identity_attack": 0.0,
            }

        results = self.model.predict(text)
        return {k: round(v, 4) for k, v in results.items()}

    def classify_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Classify a batch of texts.

        Returns:
            List of dicts, each with toxicity dimension scores.
        """
        clean = [t if t and t.strip() else "safe" for t in texts]
        results = self.model.predict(clean)

        # Detoxify returns {dim: [scores]} for batch
        n = len(clean)
        batch_results = []
        for i in range(n):
            batch_results.append(
                {k: round(v[i], 4) for k, v in results.items()}
            )
        return batch_results

    def is_toxic(self, text: str) -> bool:
        """Quick check: is the text toxic above the configured threshold?"""
        scores = self.classify(text)
        return scores.get("toxicity", 0.0) >= self.threshold

    def summarize(self, text: str) -> Dict:
        """
        Return a compact summary: overall toxicity flag + top dimension.
        """
        scores = self.classify(text)
        max_dim = max(scores, key=scores.get)
        return {
            "is_toxic": scores.get("toxicity", 0.0) >= self.threshold,
            "toxicity_score": scores.get("toxicity", 0.0),
            "top_dimension": max_dim,
            "top_score": scores[max_dim],
            "all_scores": scores,
        }
