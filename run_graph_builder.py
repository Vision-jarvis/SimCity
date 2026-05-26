import logging
import sys
from graph.graph_builder import GraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting SimCity Graph Engine...")
    
    # Initialize the GraphBuilder (connects to Kafka and Neo4j)
    builder = GraphBuilder()
    
    if builder.consumer and builder.neo4j_client.driver:
        logger.info("Graph Builder is fully connected. Starting ingestion loop.")
        builder.start_consuming()
    else:
        logger.error("Graph Builder could not start due to missing connections.")
        if not builder.consumer:
            logger.error("-> Kafka connection failed.")
        if not builder.neo4j_client.driver:
            logger.error("-> Neo4j connection failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
