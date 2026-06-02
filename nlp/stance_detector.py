"""
Stance detection using cross-encoder NLI (Natural Language Inference).
Determines whether a text supports, contradicts, or is neutral toward a claim.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StanceDetector:
    """
    Uses a cross-encoder NLI model to detect stance toward a given claim.
    Labels: entailment (support), contradiction (oppose), neutral.
    """

    STANCE_LABELS = ["contradiction", "neutral", "entailment"]
    STANCE_MAP = {
        "entailment": "support",
        "contradiction": "oppose",
        "neutral": "neutral",
    }

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        from nlp.config import STANCE_MODEL_NAME, DEVICE

        self.model_name = model_name or STANCE_MODEL_NAME
        self.device = device or DEVICE
        self._model = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                logger.info(f"Loading stance model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
                self._model.eval()
                if self.device != "cpu":
                    self._model = self._model.to(self.device)
                logger.info("Stance model loaded.")
            except ImportError:
                raise ImportError(
                    "transformers is required. Install with: pip install transformers"
                )
        return self._model

    def detect(self, text: str, claim: str) -> Dict[str, float]:
        """
        Detect stance of `text` toward `claim`.

        Args:
            text: The text whose stance to assess.
            claim: The reference claim or proposition.

        Returns:
            Dict with 'stance' (support/oppose/neutral) and 'scores'.
        """
        import torch

        inputs = self._tokenizer(
            text, claim, return_tensors="pt", truncation=True, max_length=512
        )
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        scores = {
            self.STANCE_LABELS[i]: round(probs[i].item(), 4)
            for i in range(len(self.STANCE_LABELS))
        }
        top_label = max(scores, key=scores.get)

        return {
            "stance": self.STANCE_MAP[top_label],
            "raw_label": top_label,
            "confidence": scores[top_label],
            "scores": scores,
        }

    def detect_batch(
        self, texts: List[str], claim: str
    ) -> List[Dict[str, float]]:
        """
        Detect stance for a batch of texts against the same claim.
        """
        return [self.detect(text, claim) for text in texts]

    def detect_pairwise(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Dict[str, float]]:
        """
        Detect stance for a list of (text, claim) pairs.
        """
        return [self.detect(text, claim) for text, claim in pairs]
