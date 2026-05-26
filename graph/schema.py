import logging
from graph.neo4j_client import GraphClient

logger = logging.getLogger(__name__)

def initialize_schema(client: GraphClient):
    """
    Creates uniqueness constraints to prevent duplicate nodes.
    Requires Neo4j 5.x syntax.
    """
    logger.info("Initializing Neo4j schema constraints...")

    constraints = [
        # Authors should be unique by their name (username/domain)
        "CREATE CONSTRAINT author_name_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
        
        # Platforms should be unique by their integer ID
        "CREATE CONSTRAINT platform_id_unique IF NOT EXISTS FOR (p:Platform) REQUIRE p.id IS UNIQUE",
        
        # Distinct Event nodes should be unique by their string ID
        "CREATE CONSTRAINT redditpost_id_unique IF NOT EXISTS FOR (e:RedditPost) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT hnstory_id_unique IF NOT EXISTS FOR (e:HNStory) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT gdeltnews_id_unique IF NOT EXISTS FOR (e:GDELTNews) REQUIRE e.id IS UNIQUE",
    ]

    for query in constraints:
        try:
            client.execute_write(query)
            logger.debug(f"Executed: {query}")
        except Exception as e:
            logger.error(f"Failed to execute schema constraint: {query}\nError: {e}")

    logger.info("Schema initialization complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db_client = GraphClient()
    if db_client.driver:
        initialize_schema(db_client)
        db_client.close()
    else:
        logger.error("Cannot initialize schema: Neo4j offline.")
