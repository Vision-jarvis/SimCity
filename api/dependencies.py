"""FastAPI dependency injection for shared services."""

import logging
from typing import Optional
from graph.neo4j_client import GraphClient

logger = logging.getLogger(__name__)

# Singleton instances
_graph_client: Optional[GraphClient] = None


def get_graph_client() -> Optional[GraphClient]:
    """Get or create a shared Neo4j client."""
    global _graph_client
    if _graph_client is None:
        try:
            _graph_client = GraphClient()
            if _graph_client.driver is None:
                logger.warning("Neo4j is not available. Graph endpoints will return mock data.")
                _graph_client = None
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
            _graph_client = None
    return _graph_client
