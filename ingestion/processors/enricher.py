"""
Event enricher — adds NLP metadata (sentiment, toxicity, embeddings, misinfo score)
to normalized events before graph ingestion.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class Enricher:
    """
    Enriches events with NLP analysis results. Modules are loaded lazily
    so this class works even if heavy ML dependencies are not installed.
    """

    def __init__(self, enable_embeddings: bool = True, enable_sentiment: bool = True,
                 enable_toxicity: bool = True, enable_misinfo: bool = True):
        self.enable_embeddings = enable_embeddings
        self.enable_sentiment = enable_sentiment
        self.enable_toxicity = enable_toxicity
        self.enable_misinfo = enable_misinfo

        # Lazy-loaded NLP modules
        self._embedding_engine = None
        self._sentiment_analyzer = None
        self._toxicity_classifier = None
        self._misinfo_scorer = None

        self.stats = {"enriched": 0, "errors": 0}

    @property
    def embedding_engine(self):
        if self._embedding_engine is None and self.enable_embeddings:
            try:
                from nlp.embeddings import EmbeddingEngine
                self._embedding_engine = EmbeddingEngine()
            except Exception as e:
                logger.warning(f"Could not load EmbeddingEngine: {e}")
                self.enable_embeddings = False
        return self._embedding_engine

    @property
    def sentiment_analyzer(self):
        if self._sentiment_analyzer is None and self.enable_sentiment:
            try:
                from nlp.sentiment_analyzer import SentimentAnalyzer
                self._sentiment_analyzer = SentimentAnalyzer()
            except Exception as e:
                logger.warning(f"Could not load SentimentAnalyzer: {e}")
                self.enable_sentiment = False
        return self._sentiment_analyzer

    @property
    def toxicity_classifier(self):
        if self._toxicity_classifier is None and self.enable_toxicity:
            try:
                from nlp.toxicity_classifier import ToxicityClassifier
                self._toxicity_classifier = ToxicityClassifier()
            except Exception as e:
                logger.warning(f"Could not load ToxicityClassifier: {e}")
                self.enable_toxicity = False
        return self._toxicity_classifier

    @property
    def misinfo_scorer(self):
        if self._misinfo_scorer is None and self.enable_misinfo:
            try:
                from nlp.misinformation_scorer import MisinformationScorer
                self._misinfo_scorer = MisinformationScorer()
            except Exception as e:
                logger.warning(f"Could not load MisinformationScorer: {e}")
                self.enable_misinfo = False
        return self._misinfo_scorer

    def enrich(self, event: Dict) -> Dict:
        """
        Enrich a single normalized event with NLP metadata.

        Adds 'nlp' key to the event containing:
        - sentiment: {label, score}
        - toxicity: {toxicity, severe_toxicity, ...}
        - misinfo: {check_worthy_score, classification}
        - embedding: list of floats (if enabled)
        """
        content = event.get("content", "")
        nlp_data = {}

        try:
            # Sentiment
            if self.enable_sentiment and self.sentiment_analyzer and content:
                nlp_data["sentiment"] = self.sentiment_analyzer.analyze(content)

            # Toxicity
            if self.enable_toxicity and self.toxicity_classifier and content:
                nlp_data["toxicity"] = self.toxicity_classifier.classify(content)

            # Misinformation
            if self.enable_misinfo and self.misinfo_scorer and content:
                nlp_data["misinfo"] = self.misinfo_scorer.score(content)

            # Embeddings
            if self.enable_embeddings and self.embedding_engine and content:
                emb = self.embedding_engine.encode(content)
                nlp_data["embedding"] = emb[0].tolist()

            self.stats["enriched"] += 1

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Enrichment error for event {event.get('id')}: {e}")

        event["nlp"] = nlp_data
        return event

    def enrich_batch(self, events: list) -> list:
        """Enrich a batch of events."""
        return [self.enrich(e) for e in events]

    def get_stats(self) -> Dict:
        """Return enrichment statistics."""
        return self.stats
