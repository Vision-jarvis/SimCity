"""
Text summarization using facebook/bart-large-cnn.
For condensing long posts, articles, and comment threads.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class Summarizer:
    """
    Abstractive summarization using BART-large-CNN.
    Useful for condensing GDELT articles, long Reddit posts, and comment threads.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        device: Optional[str] = None,
    ):
        from nlp.config import (
            SUMMARIZER_MODEL_NAME,
            SUMMARIZER_MAX_LENGTH,
            SUMMARIZER_MIN_LENGTH,
            DEVICE,
        )

        self.model_name = model_name or SUMMARIZER_MODEL_NAME
        self.max_length = max_length or SUMMARIZER_MAX_LENGTH
        self.min_length = min_length or SUMMARIZER_MIN_LENGTH
        self.device = device or DEVICE
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy-load the summarization pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline

                logger.info(f"Loading summarizer: {self.model_name}")
                self._pipeline = hf_pipeline(
                    "summarization",
                    model=self.model_name,
                    device=-1 if self.device == "cpu" else 0,
                )
                logger.info("Summarizer loaded.")
            except ImportError:
                raise ImportError(
                    "transformers is required. Install with: pip install transformers"
                )
        return self._pipeline

    def summarize(self, text: str) -> str:
        """
        Summarize a single text document.

        Args:
            text: Input text (up to ~1024 tokens for BART).

        Returns:
            Summary string.
        """
        if not text or len(text.strip()) < 50:
            return text  # Too short to summarize

        result = self.pipeline(
            text[:1024],  # BART max input
            max_length=self.max_length,
            min_length=self.min_length,
            do_sample=False,
        )
        return result[0]["summary_text"]

    def summarize_batch(
        self, texts: List[str], batch_size: int = 4
    ) -> List[str]:
        """
        Summarize a batch of texts.

        Returns:
            List of summary strings.
        """
        # Filter out very short texts
        processed = []
        short_indices = set()
        for i, t in enumerate(texts):
            if not t or len(t.strip()) < 50:
                short_indices.add(i)
            else:
                processed.append(t[:1024])

        if processed:
            results = self.pipeline(
                processed,
                max_length=self.max_length,
                min_length=self.min_length,
                do_sample=False,
                batch_size=batch_size,
            )
            summaries_iter = iter(r["summary_text"] for r in results)
        else:
            summaries_iter = iter([])

        output = []
        for i, t in enumerate(texts):
            if i in short_indices:
                output.append(t)
            else:
                output.append(next(summaries_iter))

        return output
