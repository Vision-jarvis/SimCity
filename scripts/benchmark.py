"""
Benchmark script — measures performance of core components.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch


def benchmark_seir():
    """Benchmark SEIR-Z-D simulation engine."""
    from simulation.engine import SimCityEngine
    from models.deffuant import SmoothDeffuant

    engine = SimCityEngine(None, None, None, None, None)
    engine.initialize_simulation(
        initial_state=[90000, 1000, 0, 9000, 0, 0],
        N=100000, theta=2.0, sigma=0.5, gamma_I=0.1, delta_D=0.01,
    )

    deffuant = SmoothDeffuant(eta=0.5, epsilon_base=0.3, kappa=10.0)
    opinions = torch.cat([
        torch.normal(0.2, 0.1, (50000,)),
        torch.normal(0.8, 0.1, (50000,)),
    ])

    start = time.time()
    for _ in range(10):
        engine.run_scenario(
            steps=30, dt=1.0, base_beta_macro=0.8,
            base_lambda=50.0, baseline_lambda=1.0,
            decay_gamma=0.2, phi=0.05,
            deffuant_model=deffuant, initial_opinions=opinions,
            content_opinion=0.9,
        )
    elapsed = time.time() - start
    print(f"SEIR-Z-D (30 steps × 10 runs): {elapsed:.3f}s ({elapsed/10*1000:.1f}ms/run)")


def benchmark_dedup():
    """Benchmark deduplication throughput."""
    from ingestion.processors.deduplicator import Deduplicator

    dedup = Deduplicator()
    events = [{"id": f"evt_{i}", "content": f"Content {i}"} for i in range(100000)]

    start = time.time()
    for e in events:
        dedup.process(e)
    elapsed = time.time() - start
    print(f"Deduplicator (100k events): {elapsed:.3f}s ({100000/elapsed:.0f} events/sec)")


def benchmark_community():
    """Benchmark community detection."""
    import networkx as nx
    from graph.community_detector import CommunityDetector

    G = nx.barabasi_albert_graph(5000, 3)
    detector = CommunityDetector()

    start = time.time()
    detector.detect_louvain(G)
    elapsed = time.time() - start
    sizes = detector.get_community_sizes()
    print(f"Louvain (5k nodes): {elapsed:.3f}s, {len(sizes)} communities")


def main():
    print("=" * 50)
    print("SimCity Performance Benchmarks")
    print("=" * 50)

    print("\n--- SEIR-Z-D Engine ---")
    benchmark_seir()

    print("\n--- Deduplicator ---")
    benchmark_dedup()

    print("\n--- Community Detection ---")
    benchmark_community()

    print("\n" + "=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
