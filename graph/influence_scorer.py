"""
Dynamic influence scoring for graph nodes.
Combines Temporal PageRank, engagement velocity, and degree centrality.
"""

import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)


class GraphInfluenceScorer:
    """
    Computes the Dynamic Influence Score I(v,t) from the whitepaper:
    I(v,t) = ω₁·TPR(v,t) + ω₂·dE(v,t)/dt + ω₃·C_D(v,t)

    Where:
    - TPR = Temporal PageRank
    - dE/dt = Engagement velocity
    - C_D = Temporal Degree Centrality
    """

    def __init__(
        self,
        omega_1: float = 0.4,
        omega_2: float = 0.3,
        omega_3: float = 0.3,
        damping: float = 0.85,
        pagerank_iterations: int = 100,
    ):
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.omega_3 = omega_3
        self.damping = damping
        self.pagerank_iterations = pagerank_iterations

        # Caches
        self._pagerank_scores: Dict[str, float] = {}
        self._engagement_history: Dict[str, list] = {}
        self._degree_cache: Dict[str, int] = {}

    def compute_pagerank(self, graph) -> Dict[str, float]:
        """
        Compute PageRank on the graph using NetworkX.

        Args:
            graph: NetworkX DiGraph.

        Returns:
            Dict mapping node_id → PageRank score.
        """
        import networkx as nx

        if len(graph) == 0:
            return {}

        pr = nx.pagerank(
            graph,
            alpha=self.damping,
            max_iter=self.pagerank_iterations,
        )
        self._pagerank_scores = pr
        return pr

    def compute_degree_centrality(self, graph) -> Dict[str, float]:
        """
        Compute degree centrality for all nodes.
        """
        import networkx as nx

        centrality = nx.degree_centrality(graph)
        self._degree_cache = {k: graph.degree(k) for k in graph.nodes()}
        return centrality

    def update_engagement(self, node_id: str, engagement: float, timestamp: float):
        """
        Record an engagement event for computing velocity.
        """
        if node_id not in self._engagement_history:
            self._engagement_history[node_id] = []

        self._engagement_history[node_id].append((timestamp, engagement))

        # Keep only last 100 events per node
        if len(self._engagement_history[node_id]) > 100:
            self._engagement_history[node_id] = self._engagement_history[node_id][-100:]

    def _compute_engagement_velocity(self, node_id: str, window: float = 3600.0) -> float:
        """
        Compute engagement velocity (dE/dt) for a node over a time window.

        Args:
            node_id: The node to compute velocity for.
            window: Time window in seconds (default: 1 hour).

        Returns:
            Engagement events per second within the window.
        """
        history = self._engagement_history.get(node_id, [])
        if len(history) < 2:
            return 0.0

        now = time.time()
        recent = [e for t, e in history if now - t <= window]

        if not recent:
            return 0.0

        return sum(recent) / window

    def score(self, node_id: str) -> float:
        """
        Compute the Dynamic Influence Score for a single node.
        Requires that compute_pagerank has been called recently.
        """
        tpr = self._pagerank_scores.get(node_id, 0.0)
        velocity = self._compute_engagement_velocity(node_id)
        degree = self._degree_cache.get(node_id, 0)

        # Normalize degree to [0, 1] range
        max_degree = max(self._degree_cache.values()) if self._degree_cache else 1
        degree_norm = degree / max(max_degree, 1)

        return (
            self.omega_1 * tpr
            + self.omega_2 * min(velocity, 1.0)  # Cap velocity contribution
            + self.omega_3 * degree_norm
        )

    def score_all(self, graph) -> Dict[str, float]:
        """
        Compute influence scores for all nodes in the graph.
        """
        self.compute_pagerank(graph)
        self.compute_degree_centrality(graph)

        scores = {}
        for node_id in graph.nodes():
            scores[node_id] = self.score(node_id)

        return scores

    def top_k(self, graph, k: int = 20) -> list:
        """
        Return the top-K most influential nodes.

        Returns:
            List of (node_id, score) tuples sorted by score descending.
        """
        scores = self.score_all(graph)
        return sorted(scores.items(), key=lambda x: -x[1])[:k]

    def write_to_neo4j(self, neo4j_client, graph):
        """
        Write influence scores to Neo4j as INFLUENCE edges and node properties.
        """
        scores = self.score_all(graph)

        for node_id, score in scores.items():
            query = """
            MATCH (a:Author {name: $node_id})
            SET a.influence_score = $score, a.influence_updated = timestamp()
            """
            try:
                neo4j_client.execute_write(query, {
                    "node_id": node_id,
                    "score": score,
                })
            except Exception as e:
                logger.error(f"Failed to write influence for {node_id}: {e}")

        logger.info(f"Updated influence scores for {len(scores)} nodes.")
