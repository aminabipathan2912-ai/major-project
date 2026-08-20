"""
backend/app/api/websocket.py
WebSocket manager for real-time dashboard updates.
Broadcasts prediction results, incidents, and alerts to all connected clients.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        logger.info("WebSocket connected. Active connections: %d", len(self._connections))
        # Send initial handshake
        await self._send_to(ws, {
            "type": "connected",
            "message": "Connected to Live Multimodal Monitoring System",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)
        logger.info("WebSocket disconnected. Active connections: %d", len(self._connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        if not self._connections:
            return
        dead: List[WebSocket] = []
        for ws in list(self._connections):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(json.dumps(message, default=str))
            except Exception as exc:
                logger.warning("Failed to send to WebSocket: %s", exc)
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)

    async def _send_to(self, ws: WebSocket, message: Dict[str, Any]) -> None:
        try:
            await ws.send_text(json.dumps(message, default=str))
        except Exception as exc:
            logger.warning("Failed to send to WebSocket: %s", exc)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Global singleton — imported by routes and services that need to broadcast
manager = ConnectionManager()


@router.websocket("/ws/monitor")
async def monitor_ws(websocket: WebSocket) -> None:
    """
    Real-time monitoring WebSocket endpoint.
    Clients receive:
      - type: "connected"         — on connection
      - type: "prediction"        — new modality prediction
      - type: "fusion_result"     — fusion engine output
      - type: "incident"          — new incident created
      - type: "alert"             — new alert fired
      - type: "ping"              — keepalive
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; server pushes data via broadcast()
            data = await websocket.receive_text()
            # Handle client pings
            if data == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        manager.disconnect(websocket)
