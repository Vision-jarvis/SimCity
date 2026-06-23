import os

# python-dotenv is optional; environment variables still work without it
# (and without a .env file), so imports never hard-fail in minimal envs/CI.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Kafka Configuration
KAFKA_BROKER_URL = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "simcity_events")

# Reddit API Credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "simcity_digital_twin/1.0")

# Hacker News Configuration
HN_API_BASE_URL = "https://hacker-news.firebaseio.com/v0"

# Wikipedia / Wikimedia Configuration (free, no key required)
WIKIPEDIA_API_URL = os.getenv(
    "WIKIPEDIA_API_URL", "https://en.wikipedia.org/w/api.php"
)

# Bluesky (AT Protocol) — free public reads, no key required.
# The public AppView serves unauthenticated reads. Optionally set an app
# password (https://bsky.app/settings/app-passwords) for higher rate limits.
BLUESKY_APPVIEW_URL = os.getenv("BLUESKY_APPVIEW_URL", "https://public.api.bsky.app")
BLUESKY_PDS_URL = os.getenv("BLUESKY_PDS_URL", "https://bsky.social")
BLUESKY_IDENTIFIER = os.getenv("BLUESKY_IDENTIFIER", "")  # e.g. you.bsky.social (optional)
BLUESKY_APP_PASSWORD = os.getenv("BLUESKY_APP_PASSWORD", "")  # app password (optional)
# Public mode (no key) follows these accounts via app.bsky.feed.getAuthorFeed.
BLUESKY_ACTORS = [
    a.strip()
    for a in os.getenv("BLUESKY_ACTORS", "bsky.app,nytimes.com,bbc.com").split(",")
    if a.strip()
]
# Search (app.bsky.feed.searchPosts) requires auth — used only when an app
# password is configured.
BLUESKY_QUERIES = [
    q.strip()
    for q in os.getenv("BLUESKY_QUERIES", "misinformation,breaking news").split(",")
    if q.strip()
]

# Target Data Schema
# All events pushed to Kafka will conform to this basic dictionary
"""
{
    "id": str,                  # Unique identifier
    "platform": int,            # 0=Reddit 1=HN 2=GDELT 3=RSS 4=YouTube 5=Wikipedia 6=Bluesky
    "timestamp": float,         # Unix timestamp in seconds
    "author": str,              # Username or domain
    "content": str,             # Post title, comment body, or news abstract
    "metadata": dict            # Any extra platform-specific fields
}
"""
