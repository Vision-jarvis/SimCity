# Data Sources

## Overview

SimCity ingests from 5 real-time internet data sources, each mapped to a platform ID.

| Platform | ID | API | Free Tier | Ingester |
|----------|------|------|-----------|----------|
| Reddit | 0 | PRAW (OAuth) | Yes (60 req/min) | `ingestion/sources/reddit_ingester.py` |
| Hacker News | 1 | Firebase API | Yes (unlimited) | `ingestion/sources/hackernews_ingester.py` |
| GDELT | 2 | GDELT Project | Yes (unlimited) | `ingestion/sources/gdelt_ingester.py` |
| RSS | 3 | feedparser | Yes (N/A) | `ingestion/sources/rss_ingester.py` |
| YouTube | 4 | Data API v3 | Yes (10k units/day) | `ingestion/sources/youtube_ingester.py` |

## Event Schema

All events are normalized to:

```json
{
  "id": "platform_uniqueId",
  "platform": 0,
  "timestamp": 1700000000.0,
  "author": "username",
  "content": "text content",
  "metadata": { ... }
}
```

## Reddit
- **Source**: r/all, configurable subreddits
- **Fields**: title, selftext, score, num_comments, subreddit
- **Rate limit**: 60 requests/minute with OAuth

## Hacker News
- **Source**: Firebase API (newstories, topstories)
- **Fields**: title, url, score, descendants, type
- **Rate limit**: None (public Firebase endpoint)

## GDELT
- **Source**: GDELT 2.0 Events Database
- **Fields**: title, tone, themes, goldstein_scale, source_country
- **Rate limit**: None

## RSS
- **Source**: Configurable feed URLs (NYT, BBC, etc.)
- **Fields**: title, summary, tags, link
- **Rate limit**: Based on feed poll interval

## YouTube
- **Source**: YouTube Data API v3
- **Fields**: title, description, view_count, like_count, channel
- **Rate limit**: 10,000 quota units/day (search=100, details=1)
