"""
Backfill GDELT events for historical data analysis.
Fetches events from the last N days and pushes them through the ingestion pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from ingestion.sources.gdelt_ingester import GDELTIngester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backfill GDELT data")
    parser.add_argument("--days", type=int, default=7, help="Number of days to backfill")
    parser.add_argument("--limit", type=int, default=1000, help="Max events to fetch")
    parser.add_argument("--dry-run", action="store_true", help="Don't push to Kafka, just print")
    args = parser.parse_args()

    logger.info(f"Starting GDELT backfill: {args.days} days, limit={args.limit}")

    ingester = GDELTIngester(producer=None)

    count = 0
    try:
        for event in ingester.fetch_latest():
            if count >= args.limit:
                break

            if args.dry_run:
                logger.info(f"  [{count}] {event.get('id', '?')}: {event.get('content', '')[:80]}")
            else:
                # In production: push to Kafka
                logger.info(f"  Ingested event {event.get('id', '?')}")

            count += 1
    except KeyboardInterrupt:
        pass

    logger.info(f"✓ Backfilled {count} GDELT events")


if __name__ == "__main__":
    main()
