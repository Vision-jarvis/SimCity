import logging
import threading
import time
import sys

from ingestion.kafka_producer import EventProducer
from ingestion.sources.reddit_ingester import RedditIngester
from ingestion.sources.hackernews_ingester import HackerNewsIngester
from ingestion.sources.gdelt_ingester import GDELTIngester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Data Ingestion Layer")
    
    # Initialize the Kafka producer
    # (Will fall back to dry-run mode if Kafka is offline)
    producer = EventProducer()

    ingesters = []
    
    try:
        # 1. Start HackerNews Ingester
        hn_ingester = HackerNewsIngester(producer, poll_interval=10)
        ingesters.append(hn_ingester)
        
        # 2. Start GDELT Ingester
        gdelt_ingester = GDELTIngester(producer, poll_interval=300)
        ingesters.append(gdelt_ingester)
        
        # 3. Start Reddit Ingester (Requires API keys)
        try:
            reddit_ingester = RedditIngester(producer, subreddits=['worldnews', 'news', 'technology'])
            ingesters.append(reddit_ingester)
        except Exception as e:
            logger.warning(f"Skipping Reddit ingester: {e}")

        # Start all ingesters in separate threads
        threads = []
        for ingester in ingesters:
            t = threading.Thread(target=ingester.start_stream, daemon=True)
            t.start()
            threads.append(t)
            
        logger.info("All ingesters started. Press Ctrl+C to stop.")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down ingesters...")
        for ingester in ingesters:
            ingester.stop_stream()
        producer.flush()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
