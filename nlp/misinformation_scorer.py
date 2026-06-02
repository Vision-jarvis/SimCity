"""
Misinformation scoring using the ClaimBuster API (free tier).
Scores claims for check-worthiness.
"""

import logging
import requests
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MisinformationScorer:
    """
    Scores text for check-worthiness (likelihood of containing
    factual claims worth verifying) using the ClaimBuster API.

    Falls back to a simple heuristic scorer if the API is unavailable.
    """

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        from nlp.config import CLAIMBUSTER_API_KEY, CLAIMBUSTER_API_URL

        self.api_key = api_key or CLAIMBUSTER_API_KEY
        self.api_url = api_url or CLAIMBUSTER_API_URL
        self.session = requests.Session()

    def score(self, text: str) -> Dict[str, float]:
        """
        Score a single text for misinformation risk.

        Returns:
            Dict with 'check_worthy_score' (0-1), 'classification', and 'sentences'.
        """
        if not text or not text.strip():
            return {
                "check_worthy_score": 0.0,
                "classification": "not_check_worthy",
                "sentences": [],
            }

        # Try ClaimBuster API first
        if self.api_key:
            try:
                return self._score_api(text)
            except Exception as e:
                logger.warning(f"ClaimBuster API failed, falling back to heuristic: {e}")

        # Fallback to heuristic
        return self._score_heuristic(text)

    def _score_api(self, text: str) -> Dict:
        """Score using ClaimBuster API."""
        headers = {"x-api-key": self.api_key}
        payload = {"input_text": text[:5000]}

        resp = self.session.post(
            self.api_url, json=payload, headers=headers, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return {
                "check_worthy_score": 0.0,
                "classification": "not_check_worthy",
                "sentences": [],
            }

        # Aggregate sentence-level scores
        sentence_scores = []
        for r in results:
            sentence_scores.append({
                "text": r.get("text", ""),
                "score": round(r.get("score", 0.0), 4),
            })

        max_score = max(s["score"] for s in sentence_scores)
        avg_score = sum(s["score"] for s in sentence_scores) / len(sentence_scores)

        classification = "check_worthy" if max_score >= 0.5 else "not_check_worthy"

        return {
            "check_worthy_score": round(avg_score, 4),
            "max_sentence_score": round(max_score, 4),
            "classification": classification,
            "sentences": sentence_scores,
        }

    def _score_heuristic(self, text: str) -> Dict:
        """
        Simple heuristic fallback for misinformation signal detection.
        Checks for common misinformation indicators.
        """
        text_lower = text.lower()

        # Indicator keywords that correlate with check-worthiness
        indicators = [
            "breaking", "leaked", "exposed", "secret", "coverup", "cover-up",
            "they don't want you to know", "wake up", "mainstream media",
            "big pharma", "deep state", "false flag", "hoax", "scam",
            "conspiracy", "censored", "suppressed", "banned",
            "100%", "proven", "scientifically proven", "doctors hate",
            "miracle", "cure", "exposed the truth",
        ]

        # Emotional amplifiers
        amplifiers = [
            "!!!", "???", "BREAKING", "ALERT", "URGENT", "SHOCKING",
            "BOMBSHELL", "EXPLOSIVE",
        ]

        indicator_count = sum(1 for kw in indicators if kw in text_lower)
        amplifier_count = sum(1 for amp in amplifiers if amp in text)

        # Normalize score to [0, 1]
        raw_score = (indicator_count * 0.15 + amplifier_count * 0.1)
        score = min(1.0, raw_score)

        return {
            "check_worthy_score": round(score, 4),
            "max_sentence_score": round(score, 4),
            "classification": "check_worthy" if score >= 0.5 else "not_check_worthy",
            "sentences": [],
            "method": "heuristic",
        }

    def score_batch(self, texts: List[str]) -> List[Dict]:
        """Score a batch of texts."""
        return [self.score(t) for t in texts]
