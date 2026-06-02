"""WebSocket endpoint for real-time event streaming."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import asyncio
import time
import random
import logging

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])

# Connected clients registry
connected_clients: list = []


@router.websocket("/ws/stream")
async def stream_events(websocket: WebSocket):
    """
    Real-time event stream via WebSocket.
    Sends simulated internet events to connected clients.
    In production, this consumes from the Kafka topic.
    """
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(connected_clients)}")

    try:
        while True:
            # In production: consume from Kafka and forward
            # For now: generate synthetic stream events
            event = _generate_mock_event()
            await websocket.send_json(event)
            await asyncio.sleep(random.uniform(0.5, 3.0))

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(connected_clients)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)


@router.websocket("/ws/simulation/{run_id}")
async def stream_simulation(websocket: WebSocket, run_id: str):
    """
    Stream simulation results step-by-step for animated playback.
    """
    await websocket.accept()

    # Try to load cached simulation
    import os
    cache_path = os.path.join("simulation_cache", f"{run_id}.json")
    if not os.path.exists(cache_path):
        await websocket.send_json({"error": "Simulation not found"})
        await websocket.close()
        return

    with open(cache_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])

    try:
        for step in results:
            await websocket.send_json({"type": "step", "data": step})
            await asyncio.sleep(0.5)  # 500ms per step

        await websocket.send_json({"type": "complete"})
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()


def _generate_mock_event() -> dict:
    """Generate a mock streaming event for demo purposes."""
    platforms = ["reddit", "hackernews", "gdelt", "rss", "youtube"]
    sentiments = ["positive", "negative", "neutral"]
    topics = [
        "AI Safety", "Climate Change", "Crypto", "Elections",
        "Open Source", "Cybersecurity", "Space Exploration",
    ]

    return {
        "type": "event",
        "timestamp": time.time(),
        "platform": random.choice(platforms),
        "topic": random.choice(topics),
        "sentiment": random.choice(sentiments),
        "virality_score": round(random.uniform(0, 1), 3),
        "toxicity": round(random.uniform(0, 0.5), 3),
        "author": f"user_{random.randint(1, 1000)}",
    }
