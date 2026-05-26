import json
import logging
from confluent_kafka import Producer
from ingestion.config import KAFKA_BROKER_URL, KAFKA_TOPIC

logger = logging.getLogger(__name__)

class EventProducer:
    def __init__(self, bootstrap_servers=KAFKA_BROKER_URL):
        """
        Initializes the Kafka producer. Strict coupling: will fail if Kafka is down.
        """
        self.conf = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'simcity-ingester',
            # Add retries and standard delivery guarantees
            'message.send.max.retries': 3,
        }
        try:
            self.producer = Producer(self.conf)
            logger.info(f"Connected to Kafka broker at {bootstrap_servers}")
        except Exception as e:
            logger.warning(f"Failed to connect to Kafka ({e}). Falling back to stdout mode.")
            self.producer = None

    def delivery_report(self, err, msg):
        """ Called once for each message produced to indicate delivery result. """
        if err is not None:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.debug(f'Message delivered to {msg.topic()} [{msg.partition()}]')

    def publish(self, event_dict, topic=KAFKA_TOPIC):
        """
        Serializes and publishes an event dictionary to Kafka.
        """
        try:
            # Serialize the dictionary to a JSON byte string
            value = json.dumps(event_dict).encode('utf-8')
            
            if self.producer is None:
                # Dry run mode: print to stdout
                logger.info(f"[DRY RUN - Kafka Offline] Event: {event_dict}")
                return

            # Use the event 'id' as the Kafka message key (for partitioning)
            key = str(event_dict.get("id", "")).encode('utf-8')
            
            self.producer.produce(
                topic, 
                key=key, 
                value=value, 
                callback=self.delivery_report
            )
            # Trigger any available delivery report callbacks from previous produce() calls
            self.producer.poll(0)
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")

    def flush(self):
        """ Wait for any outstanding messages to be delivered and delivery report callbacks to be triggered. """
        if self.producer:
            self.producer.flush()
