import json
import logging
from confluent_kafka import Consumer, KafkaError
from graph.config import KAFKA_BROKER_URL, KAFKA_TOPIC
from graph.neo4j_client import GraphClient
from graph.schema import initialize_schema

logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    Consumes unified events from Kafka and maps them into the Neo4j knowledge graph.
    Creates distinct node types (RedditPost, HNStory, GDELTNews) based on the platform.
    """

    # Mapping Platform ID to Node Label
    PLATFORM_LABELS = {
        0: "RedditPost",
        1: "HNStory",
        2: "GDELTNews"
    }

    def __init__(self, broker_url=KAFKA_BROKER_URL, topic=KAFKA_TOPIC):
        self.topic = topic
        self.neo4j_client = GraphClient()
        
        # Initialize schema if Neo4j is online
        if self.neo4j_client.driver:
            initialize_schema(self.neo4j_client)

        self.conf = {
            'bootstrap.servers': broker_url,
            'group.id': 'graph-builder-group',
            'auto.offset.reset': 'earliest'
        }
        
        try:
            self.consumer = Consumer(self.conf)
            self.consumer.subscribe([self.topic])
            logger.info(f"Subscribed to Kafka topic '{self.topic}' at {broker_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Kafka ({e}). Running in offline mode.")
            self.consumer = None

        self.is_running = False

    def process_event(self, event):
        """
        Takes a standardized event dictionary and executes a Cypher MERGE statement.
        """
        platform_id = int(event.get("platform", -1))
        node_label = self.PLATFORM_LABELS.get(platform_id)
        
        if not node_label:
            logger.warning(f"Unknown platform ID {platform_id}. Skipping event: {event.get('id')}")
            return

        # Parameterized Cypher query using the distinct node type
        # We use MERGE to prevent duplicating nodes if we process the same event twice
        cypher_query = f"""
        // 1. Ensure the Platform node exists
        MERGE (p:Platform {{id: $platform_id}})
        
        // 2. Ensure the Author node exists
        MERGE (a:Author {{name: $author}})
        
        // 3. Ensure the specific Event node exists
        MERGE (e:{node_label} {{id: $event_id}})
        ON CREATE SET e.timestamp = $timestamp, 
                      e.content = $content,
                      e.metadata = $metadata
        ON MATCH SET e.timestamp = $timestamp, 
                     e.content = $content,
                     e.metadata = $metadata
                     
        // 4. Create the relationships
        MERGE (a)-[:POSTED]->(e)
        MERGE (e)-[:OCCURRED_ON]->(p)
        """

        parameters = {
            "platform_id": platform_id,
            "author": str(event.get("author", "[unknown]")),
            "event_id": str(event.get("id")),
            "timestamp": float(event.get("timestamp", 0.0)),
            "content": str(event.get("content", "")),
            "metadata": json.dumps(event.get("metadata", {})) # Store nested dicts as strings
        }

        try:
            self.neo4j_client.execute_write(cypher_query, parameters)
            logger.debug(f"Ingested {node_label} {event.get('id')} into Neo4j")
        except Exception as e:
            logger.error(f"Failed to ingest event {event.get('id')}: {e}")

    def start_consuming(self):
        """
        Continuous loop to poll Kafka and process events.
        """
        if not self.consumer:
            logger.error("Cannot start consuming: Kafka Consumer offline.")
            return

        self.is_running = True
        logger.info("Starting Kafka consumer loop for Graph Builder...")
        
        try:
            while self.is_running:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                        break

                try:
                    event_dict = json.loads(msg.value().decode('utf-8'))
                    self.process_event(event_dict)
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON from Kafka message")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.stop_consuming()

    def stop_consuming(self):
        self.is_running = False
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed.")
        if self.neo4j_client:
            self.neo4j_client.close()
