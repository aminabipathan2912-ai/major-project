from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def _now() -> float:
    return time.time()


class IncidentStore:
    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    location TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp_epoch_s REAL NOT NULL,
                    status TEXT NOT NULL,
                    message_text TEXT NOT NULL DEFAULT '',
                    audio_filename TEXT NOT NULL DEFAULT '',
                    speech_result TEXT NOT NULL DEFAULT '',
                    acknowledged_at_epoch_s REAL,
                    created_at_epoch_s REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS emergency_calls (
                    id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    to_number TEXT NOT NULL,
                    twilio_sid TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at_epoch_s REAL NOT NULL,
                    FOREIGN KEY(incident_id) REFERENCES incidents(id)
                );
                """
            )
            self._conn.commit()

    def create_incident(
        self,
        *,
        event_type: str,
        camera_id: str,
        location: str,
        confidence: float,
        timestamp_epoch_s: float,
        message_text: str,
    ) -> dict[str, Any]:
        incident_id = f"INC-{uuid.uuid4().hex[:10].upper()}"
        row = {
            "id": incident_id,
            "event_type": event_type,
            "camera_id": camera_id,
            "location": location,
            "confidence": float(confidence),
            "timestamp_epoch_s": float(timestamp_epoch_s),
            "status": "DETECTED",
            "message_text": message_text,
            "audio_filename": "",
            "speech_result": "",
            "acknowledged_at_epoch_s": None,
            "created_at_epoch_s": _now(),
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO incidents (
                    id, event_type, camera_id, location, confidence, timestamp_epoch_s,
                    status, message_text, audio_filename, speech_result,
                    acknowledged_at_epoch_s, created_at_epoch_s
                ) VALUES (
                    :id, :event_type, :camera_id, :location, :confidence, :timestamp_epoch_s,
                    :status, :message_text, :audio_filename, :speech_result,
                    :acknowledged_at_epoch_s, :created_at_epoch_s
                )
                """,
                row,
            )
            self._conn.commit()
        return row

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def latest_incident(self) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM incidents ORDER BY created_at_epoch_s DESC LIMIT 1"
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def list_incidents(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM incidents ORDER BY created_at_epoch_s DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def update_incident(self, incident_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [incident_id]
        with self._lock:
            self._conn.execute(f"UPDATE incidents SET {assignments} WHERE id = ?", values)
            self._conn.commit()

    def create_call(
        self, *, incident_id: str, role: str, to_number: str
    ) -> dict[str, Any]:
        call_id = f"CALL-{uuid.uuid4().hex[:10].upper()}"
        row = {
            "id": call_id,
            "incident_id": incident_id,
            "role": role,
            "to_number": to_number,
            "twilio_sid": "",
            "status": "queued",
            "attempts": 0,
            "last_error": "",
            "updated_at_epoch_s": _now(),
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO emergency_calls (
                    id, incident_id, role, to_number, twilio_sid, status,
                    attempts, last_error, updated_at_epoch_s
                ) VALUES (
                    :id, :incident_id, :role, :to_number, :twilio_sid, :status,
                    :attempts, :last_error, :updated_at_epoch_s
                )
                """,
                row,
            )
            self._conn.commit()
        return row

    def update_call(self, call_id: str, **fields: Any) -> None:
        fields["updated_at_epoch_s"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [call_id]
        with self._lock:
            self._conn.execute(f"UPDATE emergency_calls SET {assignments} WHERE id = ?", values)
            self._conn.commit()

    def get_call(self, call_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM emergency_calls WHERE id = ?", (call_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def calls_for_incident(self, incident_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM emergency_calls WHERE incident_id = ? ORDER BY updated_at_epoch_s",
                (incident_id,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
