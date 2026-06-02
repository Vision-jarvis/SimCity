"""
Community detection using Louvain algorithm and Label Propagation
on the SimCity knowledge graph.
"""

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CommunityDetector:
    """
    Detects communities in the knowledge graph using:
    1. Louvain algorithm (modularity optimization)
    2. Label Propagation (faster, for streaming updates)

    Works with NetworkX graphs extracted from Neo4j.
    """

    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution
        self.communities: Dict[str, int] = {}
        self.modularity: float = 0.0

    def detect_louvain(self, graph) -> Dict[str, int]:
        """
        Run Louvain community detection on a NetworkX graph.

        Args:
            graph: NetworkX graph (undirected).

        Returns:
            Dict mapping node_id → community_id.
        """
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(
                graph, resolution=self.resolution
            )
            self.modularity = community_louvain.modularity(partition, graph)
            self.communities = partition

            n_communities = len(set(partition.values()))
            logger.info(
                f"Louvain detected {n_communities} communities "
                f"(modularity: {self.modularity:.4f})"
            )
            return partition

        except ImportError:
            raise ImportError(
                "python-louvain is required. Install with: pip install python-louvain"
            )

    def detect_label_propagation(self, graph) -> Dict[str, int]:
        """
        Run Label Propagation for fast community detection.
        Better for streaming/incremental updates.

        Args:
            graph: NetworkX graph.

        Returns:
            Dict mapping node_id → community_id.
        """
        import networkx as nx

        communities_gen = nx.community.label_propagation_communities(graph)
        partition = {}
        for cid, community in enumerate(communities_gen):
            for node in community:
                partition[node] = cid

        self.communities = partition
        logger.info(f"Label Propagation detected {len(set(partition.values()))} communities")
        return partition

    def get_community_sizes(self) -> Dict[int, int]:
        """Return community_id → member_count mapping."""
        sizes: Dict[int, int] = {}
        for cid in self.communities.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        return dict(sorted(sizes.items(), key=lambda x: -x[1]))

    def get_community_members(self, community_id: int) -> List[str]:
        """Return all node IDs belonging to a specific community."""
        return [
            node for node, cid in self.communities.items()
            if cid == community_id
        ]

    def get_cross_community_edges(self, graph) -> List[Tuple[str, str, int, int]]:
        """
        Find edges that cross community boundaries.
        These are key for narrative transfer detection.

        Returns:
            List of (source, target, source_community, target_community).
        """
        cross_edges = []
        for u, v in graph.edges():
            cu = self.communities.get(u, -1)
            cv = self.communities.get(v, -1)
            if cu != cv:
                cross_edges.append((u, v, cu, cv))
        return cross_edges

    def write_to_neo4j(self, neo4j_client, batch_size: int = 1000):
        """
        Write community assignments back to Neo4j as Community nodes
        and MEMBER_OF relationships.
        """
        if not self.communities:
            logger.warning("No communities detected. Run detection first.")
            return

        # Create Community nodes
        community_ids = set(self.communities.values())
        sizes = self.get_community_sizes()

        for cid in community_ids:
            query = """
            MERGE (c:Community {id: $cid})
            SET c.size = $size, c.modularity_score = $modularity, c.detected_at = timestamp()
            """
            try:
                neo4j_client.execute_write(query, {
                    "cid": cid,
                    "size": sizes.get(cid, 0),
                    "modularity": self.modularity,
                })
            except Exception as e:
                logger.error(f"Failed to write community {cid}: {e}")

        # Create MEMBER_OF relationships in batches
        items = list(self.communities.items())
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            for node_id, cid in batch:
                query = """
                MATCH (a:Author {name: $node_id})
                MATCH (c:Community {id: $cid})
                MERGE (a)-[:MEMBER_OF]->(c)
                """
                try:
                    neo4j_client.execute_write(query, {
                        "node_id": node_id,
                        "cid": cid,
                    })
                except Exception as e:
                    logger.error(f"Failed to link {node_id} to community {cid}: {e}")

        logger.info(f"Wrote {len(community_ids)} communities and {len(self.communities)} memberships to Neo4j.")
