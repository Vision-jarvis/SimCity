"""
Embedding engine using sentence-transformers for semantic similarity,
clustering, and vector search.
"""

import logging
import numpy as np
from typing import List, Union, Optional

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Wraps sentence-transformers to produce dense embeddings for text content.
    Used for: semantic search, topic clustering input, narrative similarity.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        from nlp.config import EMBEDDING_MODEL_NAME, DEVICE

        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.device = device or DEVICE
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info("Embedding model loaded successfully.")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: "
                    "pip install sentence-transformers"
                )
        return self._model

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode text(s) into dense embeddings.

        Args:
            texts: Single string or list of strings to embed.
            batch_size: Batch size for encoding.
            normalize: If True, L2-normalize embeddings (useful for cosine similarity).
            show_progress: Show progress bar during encoding.

        Returns:
            np.ndarray of shape (n_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )
        return embeddings

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        emb = self.encode([text_a, text_b], normalize=True)
        return float(np.dot(emb[0], emb[1]))

    def batch_similarity(
        self, queries: List[str], corpus: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix between queries and corpus.

        Returns:
            np.ndarray of shape (len(queries), len(corpus)).
        """
        q_emb = self.encode(queries, normalize=True)
        c_emb = self.encode(corpus, normalize=True)
        return q_emb @ c_emb.T

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
