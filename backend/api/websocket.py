"""WebSocket endpoint for real-time telemetry feed (Phase 3).

Consumes events from Redis stream ``stream:telemetry_events`` and broadcasts
to connected WebSocket clients. Supports auto-reconnect and typed events.

Event types:
  • QUARANTINE_EXECUTED — agent quarantined, memories excised
  • QUARANTINE_ROLLED_BACK — clean recovery after manual review
  • TOOL_CALL — every tool call (for live DAG updates)
  • MEMORY_WRITE — memory ingestion events
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import redis.asyncio as redis

router = APIRouter(prefix="/ws", tags=["websocket"])

TELEMETRY_STREAM = "stream:telemetry_events"


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Send message to all connected clients."""
        if not self.active_connections:
            return
        dead: List[WebSocket] = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Redis stream consumer (background task)
# ---------------------------------------------------------------------------

async def _consume_telemetry_stream(redis_client: redis.Redis) -> None:
    """Continuously read from Redis stream and broadcast to WebSocket clients."""
    last_id = "0-0"  # start from beginning
    while True:
        try:
            # Block for up to 5 seconds waiting for new events
            entries = await redis_client.xread(
                {TELEMETRY_STREAM: last_id},
                count=10,
                block=5000,
            )
            for stream, messages in entries:
                for msg_id, msg_data in messages:
                    last_id = msg_id
                    event_type = msg_data.get("event_type", "UNKNOWN")
                    payload = msg_data.get("payload", "{}")
                    timestamp = msg_data.get("timestamp", "")
                    try:
                        payload_dict = json.loads(payload)
                    except json.JSONDecodeError:
                        payload_dict = {"raw": payload}

                    await manager.broadcast({
                        "event_type": event_type,
                        "timestamp": timestamp,
                        "payload": payload_dict,
                    })
        except asyncio.CancelledError:
            break
        except Exception:
            # Log and continue — don't let stream errors kill the consumer
            await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/telemetry")
async def telemetry_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time telemetry events.

    Clients connect here to receive live:
      - Quarantine executions
      - Rollbacks
      - Tool calls (for DAG animation)
      - Memory writes
    """
    await manager.connect(websocket)
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "event_type": "CONNECTED",
            "timestamp": "",
            "payload": {"message": "Telemetry stream connected"},
        })
        # Keep connection alive — listen for client messages (ping/pong)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Startup task registration (called from main.py)
# ---------------------------------------------------------------------------

async def start_telemetry_consumer(redis_client: redis.Redis) -> asyncio.Task:
    """Start the background Redis stream consumer task."""
    return asyncio.create_task(_consume_telemetry_stream(redis_client))