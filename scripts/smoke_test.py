"""
Smoke test — quick validation that all major components are importable and functional.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PASS = "✓"
FAIL = "✗"


def check(name, func):
    try:
        func()
        print(f"  {PASS} {name}")
        return True
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        return False


def main():
    print("=" * 50)
    print("SimCity Smoke Test")
    print("=" * 50)
    results = []

    print("\n--- Core Imports ---")
    results.append(check("numpy", lambda: __import__("numpy")))
    results.append(check("torch", lambda: __import__("torch")))
    results.append(check("pandas", lambda: __import__("pandas")))
    results.append(check("networkx", lambda: __import__("networkx")))

    print("\n--- Models ---")
    results.append(check("TGN Core", lambda: __import__("models.tgn_core")))
    results.append(check("Hawkes", lambda: __import__("models.hawkes")))
    results.append(check("Virality Head", lambda: __import__("models.virality_head")))
    results.append(check("HMF Bridge", lambda: __import__("models.hmf_bridge")))
    results.append(check("Deffuant", lambda: __import__("models.deffuant")))
    results.append(check("Influence", lambda: __import__("models.influence")))
    results.append(check("Loss", lambda: __import__("models.loss")))

    print("\n--- Simulation ---")
    results.append(check("Engine", lambda: __import__("simulation.engine")))
    results.append(check("SEIR-Z-D", lambda: __import__("simulation.seir_z_d")))
    results.append(check("Agents", lambda: __import__("simulation.agents.personas")))
    results.append(check("Scenarios", lambda: __import__("simulation.scenarios.presets")))

    print("\n--- NLP ---")
    results.append(check("Embeddings", lambda: __import__("nlp.embeddings")))
    results.append(check("Sentiment", lambda: __import__("nlp.sentiment_analyzer")))
    results.append(check("Toxicity", lambda: __import__("nlp.toxicity_classifier")))
    results.append(check("Stance", lambda: __import__("nlp.stance_detector")))
    results.append(check("MisInfo", lambda: __import__("nlp.misinformation_scorer")))
    results.append(check("Summarizer", lambda: __import__("nlp.summarizer")))
    results.append(check("Topics", lambda: __import__("nlp.topic_extractor")))

    print("\n--- Graph ---")
    results.append(check("Node Types", lambda: __import__("graph.node_types")))
    results.append(check("Edge Types", lambda: __import__("graph.edge_types")))
    results.append(check("Community Detector", lambda: __import__("graph.community_detector")))
    results.append(check("Influence Scorer", lambda: __import__("graph.influence_scorer")))
    results.append(check("Graph Builder", lambda: __import__("graph.graph_builder")))

    print("\n--- Ingestion ---")
    results.append(check("Deduplicator", lambda: __import__("ingestion.processors.deduplicator")))
    results.append(check("Normalizer", lambda: __import__("ingestion.processors.normalizer")))
    results.append(check("Enricher", lambda: __import__("ingestion.processors.enricher")))

    print("\n--- ML Forecasting ---")
    results.append(check("Virality Forecaster", lambda: __import__("ml.forecasting.virality_forecaster")))
    results.append(check("Trend Predictor", lambda: __import__("ml.forecasting.trend_predictor")))
    results.append(check("Polarization Model", lambda: __import__("ml.forecasting.polarization_model")))

    print("\n--- API ---")
    results.append(check("FastAPI App", lambda: __import__("api.main")))

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("All smoke tests passed! 🎉")
    else:
        print(f"{total - passed} component(s) need attention.")
        sys.exit(1)


if __name__ == "__main__":
    main()
