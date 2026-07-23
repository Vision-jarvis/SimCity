# Data Sources

## Overview

SimCity ingests from 7 real-time internet data sources, each mapped to a platform ID.

| Platform | ID | API | Free Tier | Ingester |
|----------|------|------|-----------|----------|
| Reddit | 0 | PRAW (OAuth) | Yes (60 req/min) | `ingestion/sources/reddit_ingester.py` |
| Hacker News | 1 | Firebase API | Yes (unlimited) | `ingestion/sources/hackernews_ingester.py` |
| GDELT | 2 | GDELT Project | Yes (unlimited) | `ingestion/sources/gdelt_ingester.py` |
| RSS | 3 | feedparser | Yes (N/A) | `ingestion/sources/rss_ingester.py` |
| YouTube | 4 | Data API v3 | Yes (10k units/day) | `ingestion/sources/youtube_ingester.py` |
| Wikipedia | 5 | Wikimedia API | Yes (unlimited, no key) | `ingestion/sources/wikipedia_ingester.py` |
| Bluesky | 6 | AT Protocol AppView | Yes (no key; app password optional) | `ingestion/sources/bluesky_ingester.py` |

## Event Schema

All events are normalized to:

```json
{
  "id": "platform_uniqueId"
  "platform": 0
  "timestamp": 1700000000.0
  "author": "username"
  "content": "text content"
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

## Wikipedia
- **Source**: Wikimedia `recentchanges` API (live edit stream)
- **Fields**: title, comment, user, size_delta, type (edit/new), pageid
- **Rate limit**: None; no API key required
- **Why**: edit-volume spikes are an early signal of collective attention on breaking topics
- **Config**: `WIKIPEDIA_API_URL`

## Bluesky (AT Protocol)
- **Source**: public AppView, https://public.api.bsky.app
- **Modes**:
  - **Public (no key):** follows configured accounts via `app.bsky.feed.getAuthorFeed` (open, verified working keyless).
  - **Authenticated (optional app password):** additionally runs `app.bsky.feed.searchPosts` for query discovery, *search requires auth on Bluesky*.
- **Fields**: post text, author handle, like/repost/reply counts, post URI
- **Cost**: $0. No key for the public path. App password (optional, higher limits + search): https://bsky.app/settings/app-passwords
- **Config**: `BLUESKY_APPVIEW_URL`, `BLUESKY_PDS_URL`, `BLUESKY_ACTORS` (public)
  `BLUESKY_QUERIES` (search), optional `BLUESKY_IDENTIFIER` / `BLUESKY_APP_PASSWORD`
- **Why**: stable, key-optional social signal (the dependable free X-alternative)
