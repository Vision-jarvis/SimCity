import os

# python-dotenv is optional; environment variables still work without it.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Kafka Configuration (for Graph Builder consumer)
KAFKA_BROKER_URL = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "simcity_events")
