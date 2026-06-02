"""
Sentiment analysis using cardiffnlp/twitter-roberta-base-sentiment.
Optimized for social media text (tweets, Reddit posts, HN comments).
"""

import logging
from typing import List, Dict, Union, Optional

logger = logging.getLogger(__name__)

# Label mapping for twitter-roberta-base-sentiment
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


class SentimentAnalyzer:
    """
    Classifies text sentiment as negative / neutral / positive
    with confidence scores using the cardiffnlp twitter-roberta model.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        from nlp.config import SENTIMENT_MODEL_NAME, DEVICE

        self.model_name = model_name or SENTIMENT_MODEL_NAME
        self.device = device or DEVICE
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy-load the sentiment pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline

                logger.info(f"Loading sentiment model: {self.model_name}")
                self._pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=-1 if self.device == "cpu" else 0,
                    max_length=512,
                    truncation=True,
                )
                logger.info("Sentiment model loaded.")
            except ImportError:
                raise ImportError(
                    "transformers is required. Install with: pip install transformers"
                )
        return self._pipeline

    def analyze(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a single text.

        Returns:
            Dict with 'label' (negative/neutral/positive) and 'score' (confidence).
        """
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.0}

        result = self.pipeline(text[:512])[0]
        label = LABEL_MAP.get(result["label"], result["label"])
        return {"label": label, "score": round(result["score"], 4)}

    def analyze_batch(
        self, texts: List[str], batch_size: int = 16
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Analyze sentiment for a batch of texts.

        Returns:
            List of dicts with 'label' and 'score'.
        """
        clean_texts = [t[:512] if t and t.strip() else "neutral" for t in texts]
        results = self.pipeline(clean_texts, batch_size=batch_size)
        return [
            {
                "label": LABEL_MAP.get(r["label"], r["label"]),
                "score": round(r["score"], 4),
            }
            for r in results
        ]

    def score_numeric(self, text: str) -> float:
        """
        Return a single numeric sentiment score in [-1, 1].
        -1 = fully negative, 0 = neutral, 1 = fully positive.
        """
        result = self.analyze(text)
        mapping = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
        direction = mapping.get(result["label"], 0.0)
        return direction * result["score"]
