"""Configuration for NLP pipeline models and resources."""

import os

# === Embedding Model ===
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output dim

# === Sentiment Model ===
SENTIMENT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# === Topic Modeling ===
TOPIC_MIN_CLUSTER_SIZE = int(os.getenv("TOPIC_MIN_CLUSTER", "10"))
TOPIC_NR_TOPICS = os.getenv("TOPIC_NR_TOPICS", "auto")

# === Toxicity ===
TOXICITY_THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.5"))

# === Stance Detection ===
STANCE_MODEL_NAME = os.getenv(
    "STANCE_MODEL", "cross-encoder/nli-deberta-v3-small"
)

# === Misinformation ===
CLAIMBUSTER_API_URL = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
CLAIMBUSTER_API_KEY = os.getenv("CLAIMBUSTER_API_KEY", "")

# === Summarizer ===
SUMMARIZER_MODEL_NAME = os.getenv(
    "SUMMARIZER_MODEL", "facebook/bart-large-cnn"
)
SUMMARIZER_MAX_LENGTH = int(os.getenv("SUMMARIZER_MAX_LENGTH", "130"))
SUMMARIZER_MIN_LENGTH = int(os.getenv("SUMMARIZER_MIN_LENGTH", "30"))

# === General ===
DEVICE = os.getenv("NLP_DEVICE", "cpu")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "512"))
