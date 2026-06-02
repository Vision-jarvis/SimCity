"""
Topic extraction using BERTopic for dynamic topic modeling on streaming internet content.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class TopicExtractor:
    """
    Extracts and tracks topics from text using BERTopic.
    Supports online/incremental learning for streaming data.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        min_cluster_size: Optional[int] = None,
        nr_topics: Optional[Any] = None,
    ):
        from nlp.config import (
            EMBEDDING_MODEL_NAME,
            TOPIC_MIN_CLUSTER_SIZE,
            TOPIC_NR_TOPICS,
        )

        self.embedding_model = embedding_model or EMBEDDING_MODEL_NAME
        self.min_cluster_size = min_cluster_size or TOPIC_MIN_CLUSTER_SIZE
        self.nr_topics = nr_topics or TOPIC_NR_TOPICS
        self._model = None

    @property
    def model(self):
        """Lazy-load BERTopic model."""
        if self._model is None:
            try:
                from bertopic import BERTopic
                from sentence_transformers import SentenceTransformer

                logger.info("Initializing BERTopic model...")
                emb_model = SentenceTransformer(self.embedding_model)

                nr_topics = (
                    self.nr_topics
                    if self.nr_topics == "auto"
                    else int(self.nr_topics)
                )

                self._model = BERTopic(
                    embedding_model=emb_model,
                    min_topic_size=self.min_cluster_size,
                    nr_topics=nr_topics,
                    verbose=False,
                )
                logger.info("BERTopic model initialized.")
            except ImportError:
                raise ImportError(
                    "bertopic and sentence-transformers are required. "
                    "Install with: pip install bertopic sentence-transformers"
                )
        return self._model

    def fit_transform(
        self, documents: List[str]
    ) -> Tuple[List[int], Optional[Any]]:
        """
        Fit BERTopic on a corpus and return topic assignments.

        Args:
            documents: List of text documents.

        Returns:
            Tuple of (topic_ids, probabilities).
        """
        topics, probs = self.model.fit_transform(documents)
        logger.info(f"Extracted {len(set(topics)) - (1 if -1 in topics else 0)} topics from {len(documents)} documents.")
        return topics, probs

    def transform(self, documents: List[str]) -> Tuple[List[int], Optional[Any]]:
        """
        Assign topics to new documents using the fitted model.
        """
        return self.model.transform(documents)

    def get_topic_info(self) -> Any:
        """Return a DataFrame with topic information."""
        return self.model.get_topic_info()

    def get_topic_labels(self) -> Dict[int, str]:
        """
        Return a dict mapping topic_id to a human-readable label
        (top keywords joined).
        """
        info = self.model.get_topic_info()
        labels = {}
        for _, row in info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                labels[tid] = "outlier"
            else:
                # BERTopic stores representation as the Name column
                labels[tid] = row.get("Name", f"topic_{tid}")
        return labels

    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Return top keywords for a specific topic."""
        return self.model.get_topic(topic_id)[:top_n]

    def extract_single(self, text: str) -> Dict[str, Any]:
        """
        Extract topic for a single document.

        Returns:
            Dict with 'topic_id', 'topic_label', 'probability'.
        """
        topics, probs = self.transform([text])
        topic_id = topics[0]
        labels = self.get_topic_labels()
        return {
            "topic_id": topic_id,
            "topic_label": labels.get(topic_id, f"topic_{topic_id}"),
            "probability": float(probs[0].max()) if probs is not None else 0.0,
        }
