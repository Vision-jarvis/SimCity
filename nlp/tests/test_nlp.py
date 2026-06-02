"""Tests for NLP pipeline modules."""

import pytest


class TestSentimentAnalyzer:
    """Tests for the SentimentAnalyzer (structure-only, no model download)."""

    def test_import(self):
        from nlp.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        assert analyzer.model_name is not None

    def test_empty_text(self):
        from nlp.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("")
        assert result["label"] == "neutral"
        assert result["score"] == 0.0


class TestToxicityClassifier:
    """Tests for the ToxicityClassifier (structure-only)."""

    def test_import(self):
        from nlp.toxicity_classifier import ToxicityClassifier
        classifier = ToxicityClassifier()
        assert classifier.threshold > 0

    def test_empty_text(self):
        from nlp.toxicity_classifier import ToxicityClassifier
        classifier = ToxicityClassifier()
        result = classifier.classify("")
        assert result["toxicity"] == 0.0


class TestStanceDetector:
    """Tests for the StanceDetector (structure-only)."""

    def test_import(self):
        from nlp.stance_detector import StanceDetector
        detector = StanceDetector()
        assert detector.model_name is not None


class TestMisinformationScorer:
    """Tests for the MisinformationScorer heuristic fallback."""

    def test_import(self):
        from nlp.misinformation_scorer import MisinformationScorer
        scorer = MisinformationScorer()
        assert scorer is not None

    def test_empty_text(self):
        from nlp.misinformation_scorer import MisinformationScorer
        scorer = MisinformationScorer()
        result = scorer.score("")
        assert result["check_worthy_score"] == 0.0

    def test_heuristic_positive(self):
        from nlp.misinformation_scorer import MisinformationScorer
        scorer = MisinformationScorer()  # No API key = heuristic mode
        text = "BREAKING!!! Secret leaked documents exposed the deep state conspiracy!!!"
        result = scorer.score(text)
        assert result["check_worthy_score"] > 0.3
        assert result["method"] == "heuristic"

    def test_heuristic_negative(self):
        from nlp.misinformation_scorer import MisinformationScorer
        scorer = MisinformationScorer()
        text = "The weather today is partly cloudy with a high of 72 degrees."
        result = scorer.score(text)
        assert result["check_worthy_score"] < 0.3


class TestEmbeddingEngine:
    """Tests for the EmbeddingEngine (structure-only)."""

    def test_import(self):
        from nlp.embeddings import EmbeddingEngine
        engine = EmbeddingEngine()
        assert engine.model_name is not None


class TestTopicExtractor:
    """Tests for the TopicExtractor (structure-only)."""

    def test_import(self):
        from nlp.topic_extractor import TopicExtractor
        extractor = TopicExtractor()
        assert extractor.min_cluster_size > 0


class TestSummarizer:
    """Tests for the Summarizer (structure-only)."""

    def test_import(self):
        from nlp.summarizer import Summarizer
        s = Summarizer()
        assert s.max_length > 0

    def test_short_text_passthrough(self):
        from nlp.summarizer import Summarizer
        s = Summarizer()
        short = "Hello world."
        assert s.summarize(short) == short
