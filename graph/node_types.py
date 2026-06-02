"""
Node type definitions for the SimCity knowledge graph.
Each type maps to a Neo4j label with specific properties.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NodeType:
    """Base class for all graph node types."""
    label: str
    id_field: str
    properties: Dict[str, type] = field(default_factory=dict)

    def cypher_create(self) -> str:
        """Generate a MERGE Cypher clause template for this node type."""
        return f"MERGE (n:{self.label} {{{self.id_field}: ${self.id_field}}})"


# === Content Nodes ===

REDDIT_POST = NodeType(
    label="RedditPost",
    id_field="id",
    properties={"content": str, "timestamp": float, "score": int, "subreddit": str},
)

HN_STORY = NodeType(
    label="HNStory",
    id_field="id",
    properties={"content": str, "timestamp": float, "score": int, "type": str},
)

GDELT_NEWS = NodeType(
    label="GDELTNews",
    id_field="id",
    properties={"content": str, "timestamp": float, "goldstein_scale": float, "avg_tone": float},
)

RSS_ARTICLE = NodeType(
    label="RSSArticle",
    id_field="id",
    properties={"content": str, "timestamp": float, "feed": str, "link": str},
)

YOUTUBE_VIDEO = NodeType(
    label="YouTubeVideo",
    id_field="id",
    properties={"content": str, "timestamp": float, "view_count": int, "channel": str},
)


# === Entity Nodes ===

AUTHOR = NodeType(
    label="Author",
    id_field="name",
    properties={"platform": int, "first_seen": float, "last_seen": float},
)

TOPIC = NodeType(
    label="Topic",
    id_field="name",
    properties={"description": str, "created_at": float, "post_count": int},
)

COMMUNITY = NodeType(
    label="Community",
    id_field="id",
    properties={"name": str, "size": int, "modularity_score": float, "detected_at": float},
)

HASHTAG = NodeType(
    label="Hashtag",
    id_field="tag",
    properties={"first_seen": float, "usage_count": int},
)

ORGANIZATION = NodeType(
    label="Organization",
    id_field="name",
    properties={"type": str, "country": str, "mentions": int},
)

PLATFORM = NodeType(
    label="Platform",
    id_field="id",
    properties={"name": str},
)

NARRATIVE = NodeType(
    label="Narrative",
    id_field="id",
    properties={
        "summary": str,
        "first_seen": float,
        "platforms": list,
        "sentiment_avg": float,
        "virality_score": float,
    },
)


# === Registry ===

# Map platform ID to content node type
PLATFORM_NODE_MAP = {
    0: REDDIT_POST,
    1: HN_STORY,
    2: GDELT_NEWS,
    3: RSS_ARTICLE,
    4: YOUTUBE_VIDEO,
}

ALL_NODE_TYPES = [
    REDDIT_POST, HN_STORY, GDELT_NEWS, RSS_ARTICLE, YOUTUBE_VIDEO,
    AUTHOR, TOPIC, COMMUNITY, HASHTAG, ORGANIZATION, PLATFORM, NARRATIVE,
]
