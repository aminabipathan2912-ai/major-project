from __future__ import annotations

import json
import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row


class IncidentStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self):
        return psycopg.connect(self._database_url, row_factory=dict_row)

    def initialize(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id UUID PRIMARY KEY,
                    event_key TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    camera_id TEXT NOT NULL,
                    location TEXT NOT NULL,
                    timestamp_epoch_s DOUBLE PRECISION NOT NULL,
                    details JSONB NOT NULL DEFAULT '{}'::jsonb,
                    status TEXT NOT NULL,
                    message_text TEXT NOT NULL,
                    audio_filename TEXT,
                    speech_result TEXT,
                    acknowledged_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE TABLE IF NOT EXISTS emergency_calls (
                    id UUID PRIMARY KEY,
                    incident_id UUID NOT NULL REFERENCES incidents(id),
                    twilio_sid TEXT,
                    to_number TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )

    def create_incident(self, event: dict[str, Any], message: str) -> tuple[dict[str, Any], bool]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM incidents WHERE event_key = %s", (event["event_key"],))
            existing = cur.fetchone()
            if existing:
                return existing, False
            incident_id = uuid.uuid4()
            cur.execute(
                """
                INSERT INTO incidents (
                    id, event_key, event_type, confidence, camera_id, location,
                    timestamp_epoch_s, details, status, message_text
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, 'DETECTED', %s)
                RETURNING *
                """,
                (
                    incident_id, event["event_key"], event["event_type"], event["confidence"],
                    event["camera_id"], event["location"], event["timestamp_epoch_s"],
                    json.dumps(event.get("details", {})), message,
                ),
            )
            return cur.fetchone(), True

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM incidents WHERE id = %s", (incident_id,))
            return cur.fetchone()

    def get_active_incident(self, camera_id: str, max_age_sec: int) -> dict[str, Any] | None:
        """Return an incident that is still being handled for this camera.

        This is a server-side safety guard. Event keys include a timestamp, so
        an inference cooldown alone cannot prevent a later prediction from
        creating a second concurrent emergency call.
        """
        active_statuses = (
            "DETECTED",
            "AWAITING_ACKNOWLEDGEMENT",
            "TRIAL_CALL_STARTED",
            "CALL_ANSWERED",
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM incidents
                WHERE camera_id = %s
                  AND status = ANY(%s)
                  AND created_at >= now() - (%s * INTERVAL '1 second')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (camera_id, list(active_statuses), max(1, int(max_age_sec))),
            )
            return cur.fetchone()

    def update_incident(self, incident_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{name} = %s" for name in fields)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"UPDATE incidents SET {assignments} WHERE id = %s", (*fields.values(), incident_id))

    def create_call(self, incident_id: str, to_number: str) -> dict[str, Any]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO emergency_calls (id, incident_id, to_number, status)
                   VALUES (%s, %s, %s, 'queued') RETURNING *""",
                (uuid.uuid4(), incident_id, to_number),
            )
            return cur.fetchone()

    def update_call(self, call_id: str, **fields: Any) -> None:
        fields["updated_at"] = "now()"
        assignments = ", ".join(
            f"{name} = now()" if value == "now()" else f"{name} = %s"
            for name, value in fields.items()
        )
        values = [value for value in fields.values() if value != "now()"]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"UPDATE emergency_calls SET {assignments} WHERE id = %s", (*values, call_id))
