"""
Seed the Neo4j knowledge graph with sample data for development/testing.
Creates: platforms, sample authors, sample posts, relationships.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
from graph.neo4j_client import GraphClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLATFORMS = [
    {"id": 0, "name": "reddit"},
    {"id": 1, "name": "hackernews"},
    {"id": 2, "name": "gdelt"},
    {"id": 3, "name": "rss"},
    {"id": 4, "name": "youtube"},
]

SAMPLE_AUTHORS = [
    {"name": "tech_enthusiast", "platform": 0},
    {"name": "news_bot_42", "platform": 0},
    {"name": "pg_yc", "platform": 1},
    {"name": "dang", "platform": 1},
    {"name": "reuters", "platform": 2},
    {"name": "bbc_world", "platform": 3},
    {"name": "tech_channel", "platform": 4},
]

SAMPLE_POSTS = [
    {"id": "seed_r1", "platform": 0, "author": "tech_enthusiast", "content": "AI regulation debate heats up"},
    {"id": "seed_r2", "platform": 0, "author": "news_bot_42", "content": "Breaking: New AI policy framework proposed"},
    {"id": "seed_h1", "platform": 1, "author": "pg_yc", "content": "Show HN: Open source LLM toolkit"},
    {"id": "seed_h2", "platform": 1, "author": "dang", "content": "Ask HN: What happened to web standards?"},
    {"id": "seed_g1", "platform": 2, "author": "reuters", "content": "Climate summit yields new agreements"},
    {"id": "seed_rss1", "platform": 3, "author": "bbc_world", "content": "Global markets react to policy changes"},
    {"id": "seed_y1", "platform": 4, "author": "tech_channel", "content": "AI Safety Deep Dive - Everything You Need to Know"},
]


def seed():
    client = GraphClient()
    if client.driver is None:
        logger.error("Cannot connect to Neo4j. Make sure it's running: docker-compose up -d neo4j")
        return

    logger.info("Seeding platforms...")
    for p in PLATFORMS:
        client.execute_write(
            "MERGE (p:Platform {id: $id}) SET p.name = $name",
            p,
        )

    logger.info("Seeding authors...")
    for a in SAMPLE_AUTHORS:
        client.execute_write(
            "MERGE (a:Author {name: $name}) SET a.platform = $platform, a.first_seen = $ts",
            {**a, "ts": time.time()},
        )

    logger.info("Seeding posts...")
    label_map = {0: "RedditPost", 1: "HNStory", 2: "GDELTNews", 3: "RSSArticle", 4: "YouTubeVideo"}
    for post in SAMPLE_POSTS:
        label = label_map[post["platform"]]
        client.execute_write(
            f"MERGE (p:{label} {{id: $id}}) SET p.content = $content, p.timestamp = $ts",
            {**post, "ts": time.time()},
        )

        # POSTED relationship
        client.execute_write(
            f"MATCH (a:Author {{name: $author}}) "
            f"MATCH (p:{label} {{id: $id}}) "
            f"MERGE (a)-[:POSTED {{timestamp: $ts}}]->(p)",
            {"author": post["author"], "id": post["id"], "ts": time.time()},
        )

        # OCCURRED_ON relationship
        client.execute_write(
            f"MATCH (p:{label} {{id: $id}}) "
            f"MATCH (pl:Platform {{id: $platform}}) "
            f"MERGE (p)-[:OCCURRED_ON]->(pl)",
            {"id": post["id"], "platform": post["platform"]},
        )

    logger.info(f"✓ Seeded {len(PLATFORMS)} platforms, {len(SAMPLE_AUTHORS)} authors, {len(SAMPLE_POSTS)} posts")
    client.close()


if __name__ == "__main__":
    seed()
