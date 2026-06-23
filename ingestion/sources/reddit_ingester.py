import praw
import logging
from ingestion.sources.base import BaseIngester
from ingestion.config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

logger = logging.getLogger(__name__)

class RedditIngester(BaseIngester):
    """
    Streams new submissions and comments from specified subreddits using PRAW.
    Maps Reddit data to Platform=0.
    """
    
    def __init__(self, producer, subreddits=None):
        super().__init__(producer)
        self.subreddits_str = "+".join(subreddits) if subreddits else "all"
        
        try:
            self.reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
            self.subreddit = self.reddit.subreddit(self.subreddits_str)
            logger.info(f"Reddit ingester initialized for subreddits: {self.subreddits_str}")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client. Are credentials set? Error: {e}")
            raise

    def fetch_latest(self):
        """
        Yields a continuous stream of new submissions using PRAW's stream generator.
        """
        logger.info(f"Connecting to Reddit stream for {self.subreddits_str}")
        # Note: PRAW streams block until new items appear.
        try:
            for submission in self.subreddit.stream.submissions(skip_existing=True):
                # PRAW objects need some manual extraction to avoid infinite recursion
                metadata = {
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "subreddit": submission.subreddit.display_name,
                    "url": submission.url
                }
                
                # Yield the standard event format
                yield self.format_event(
                    event_id=f"reddit_{submission.id}",
                    platform=0, # 0 = Reddit
                    timestamp=submission.created_utc,
                    author=submission.author.name if submission.author else "[deleted]",
                    content=f"{submission.title}\n{submission.selftext}",
                    metadata=metadata
                )
        except Exception as e:
            logger.error(f"Error in Reddit stream: {e}")
