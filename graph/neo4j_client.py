import logging
from neo4j import GraphDatabase
from graph.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)

class GraphClient:
    """
    Wrapper for the Neo4j Python Driver to handle connection pooling
    and execute Cypher queries.
    """

    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j database at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {self.uri}: {e}")
            self.driver = None

    def close(self):
        if self.driver is not None:
            self.driver.close()
            logger.info("Neo4j driver closed.")

    def execute_write(self, query, parameters=None):
        """
        Executes a write transaction (CREATE, MERGE, UPDATE, DELETE).
        """
        if not self.driver:
            logger.warning("[DRY RUN] Neo4j offline. Query skipped.")
            return None

        try:
            with self.driver.session() as session:
                result = session.execute_write(self._run_query, query, parameters)
                return result
        except Exception as e:
            logger.error(f"Write transaction failed: {e}")
            raise

    def execute_read(self, query, parameters=None):
        """
        Executes a read transaction (MATCH).
        """
        if not self.driver:
            logger.warning("[DRY RUN] Neo4j offline. Query skipped.")
            return []

        try:
            with self.driver.session() as session:
                result = session.execute_read(self._run_query, query, parameters)
                return result
        except Exception as e:
            logger.error(f"Read transaction failed: {e}")
            raise

    @staticmethod
    def _run_query(tx, query, parameters):
        result = tx.run(query, parameters or {})
        return [record.data() for record in result]
