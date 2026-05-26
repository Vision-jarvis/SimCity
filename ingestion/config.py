import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Kafka Configuration
KAFKA_BROKER_URL = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "simcity_events")

# Reddit API Credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "simcity_digital_twin/1.0")

# Hacker News Configuration
HN_API_BASE_URL = "https://hacker-news.firebaseio.com/v0"

# Target Data Schema
# All events pushed to Kafka will conform to this basic dictionary
"""
{
    "id": str,                  # Unique identifier
    "platform": int,            # 0=Reddit, 1=HackerNews, 2=GDELT
    "timestamp": float,         # Unix timestamp in seconds
    "author": str,              # Username or domain
    "content": str,             # Post title, comment body, or news abstract
    "metadata": dict            # Any extra platform-specific fields
}
"""
