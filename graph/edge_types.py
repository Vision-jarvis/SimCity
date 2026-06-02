"""
Edge type definitions for the SimCity knowledge graph.
Each type maps to a Neo4j relationship with specific properties.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EdgeType:
    """Definition of a relationship type in the knowledge graph."""
    name: str
    from_label: str
    to_label: str
    properties: Dict[str, type] = field(default_factory=dict)

    def cypher_create(self) -> str:
        """Generate a MERGE Cypher clause template for this edge type."""
        return f"MERGE (a)-[r:{self.name}]->(b)"


# === Content Relationships ===

POSTED = EdgeType(
    name="POSTED",
    from_label="Author",
    to_label="*",  # Any content node
    properties={"timestamp": float},
)

OCCURRED_ON = EdgeType(
    name="OCCURRED_ON",
    from_label="*",  # Any content node
    to_label="Platform",
    properties={},
)

# === Social Relationships ===

INFLUENCE = EdgeType(
    name="INFLUENCE",
    from_label="Author",
    to_label="Author",
    properties={
        "weight": float,
        "type": str,  # direct, indirect, algorithmic
        "computed_at": float,
    },
)

REPOST = EdgeType(
    name="REPOST",
    from_label="*",
    to_label="*",  # Content reposts content
    properties={"timestamp": float, "similarity_score": float},
)

REPLY = EdgeType(
    name="REPLY",
    from_label="*",
    to_label="*",  # Content replies to content
    properties={"timestamp": float, "sentiment": str},
)

# === Semantic Relationships ===

NARRATIVE_TRANSFER = EdgeType(
    name="NARRATIVE_TRANSFER",
    from_label="*",
    to_label="*",
    properties={
        "similarity": float,
        "time_delta": float,  # seconds between events
        "source_platform": int,
        "target_platform": int,
    },
)

HAS_TOPIC = EdgeType(
    name="HAS_TOPIC",
    from_label="*",  # Any content node
    to_label="Topic",
    properties={"confidence": float},
)

TAGGED_WITH = EdgeType(
    name="TAGGED_WITH",
    from_label="*",
    to_label="Hashtag",
    properties={},
)

MENTIONS = EdgeType(
    name="MENTIONS",
    from_label="*",
    to_label="Organization",
    properties={"sentiment": str},
)

# === Community Relationships ===

MEMBER_OF = EdgeType(
    name="MEMBER_OF",
    from_label="Author",
    to_label="Community",
    properties={"since": float, "role": str},
)

PART_OF_NARRATIVE = EdgeType(
    name="PART_OF_NARRATIVE",
    from_label="*",
    to_label="Narrative",
    properties={"contribution_score": float},
)

# === Registry ===

ALL_EDGE_TYPES = [
    POSTED, OCCURRED_ON, INFLUENCE, REPOST, REPLY,
    NARRATIVE_TRANSFER, HAS_TOPIC, TAGGED_WITH, MENTIONS,
    MEMBER_OF, PART_OF_NARRATIVE,
]
