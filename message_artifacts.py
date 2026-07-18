"""Persist bounded UI artifacts against immutable chat message IDs."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeAlias

from grounding import JSONValue, UIArtifact


DatabasePath: TypeAlias = str | Path

BUSY_TIMEOUT_MS = 5_000
MAX_ARTIFACT_JSON_BYTES = 32_768
MAX_JSON_DEPTH = 8
MAX_COLLECTION_ITEMS = 200
MAX_JSON_STRING_CHARS = 12_000
MAX_JSON_KEY_CHARS = 100
MAX_BATCH_MESSAGE_IDS = 500

ALLOWED_ARTIFACT_TYPES = frozenset(
    {"source_cards", "weather_card", "search_status"}
)

_TABLE_NAME = "message_artifacts"
_REQUIRED_COLUMNS = {
    "id",
    "message_id",
    "artifact_type",
    "payload_json",
    "created_at",
}


class MessageArtifactError(RuntimeError):
    """Base class for artifact persistence errors."""


class MessageArtifactSchemaError(MessageArtifactError):
    """Raised when the current SQLite schema is incompatible."""


class MessageArtifactValidationError(MessageArtifactError, ValueError):
    """Raised when an artifact violates type or size limits."""


@dataclass(frozen=True)
class MessageArtifact:
    id: int
    message_id: int
    artifact_type: str
    payload: dict[str, JSONValue]
    created_at: str


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


def _enable_foreign_keys(conn: sqlite3.Connection) -> None:
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
        raise MessageArtifactSchemaError(
            "SQLite foreign_keys could not be enabled; initialize outside an active transaction"
        )


def _validate_parent_schema(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "chat_history"):
        raise MessageArtifactSchemaError("Required chat_history table is missing")
    columns = {
        str(row[1]): row
        for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()
    }
    if "id" not in columns or int(columns["id"][5]) != 1:
        raise MessageArtifactSchemaError("chat_history.id must be the primary key")


def _validate_artifact_schema(conn: sqlite3.Connection) -> None:
    columns = {
        str(row[1]) for row in conn.execute(f"PRAGMA table_info({_TABLE_NAME})")
    }
    missing = sorted(_REQUIRED_COLUMNS - columns)
    if missing:
        raise MessageArtifactSchemaError(
            f"message_artifacts is missing required columns: {', '.join(missing)}"
        )
    foreign_keys = conn.execute(f"PRAGMA foreign_key_list({_TABLE_NAME})").fetchall()
    has_cascade = any(
        str(row[2]) == "chat_history"
        and str(row[3]) == "message_id"
        and str(row[4]) == "id"
        and str(row[6]).upper() == "CASCADE"
        for row in foreign_keys
    )
    if not has_cascade:
        raise MessageArtifactSchemaError(
            "message_artifacts.message_id must reference chat_history.id ON DELETE CASCADE"
        )


def init_message_artifacts_schema(conn: sqlite3.Connection) -> None:
    """Create the append-only artifact table owned by ``chat_history``."""

    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be a sqlite3.Connection")
    _enable_foreign_keys(conn)
    _validate_parent_schema(conn)
    allowed_sql = ", ".join(f"'{value}'" for value in sorted(ALLOWED_ARTIFACT_TYPES))
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            artifact_type TEXT NOT NULL CHECK (artifact_type IN ({allowed_sql})),
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (message_id) REFERENCES chat_history(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{_TABLE_NAME}_message_id "
        f"ON {_TABLE_NAME}(message_id, id)"
    )
    _validate_artifact_schema(conn)


# Singular alias for callers that use the table name as a concept.
init_message_artifact_schema = init_message_artifacts_schema


def _required_message_id(message_id: object) -> int:
    if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
        raise MessageArtifactValidationError("message_id must be a positive integer")
    return message_id


def _normalize_message_ids(message_ids: Iterable[int]) -> tuple[int, ...]:
    if isinstance(message_ids, (str, bytes)):
        raise MessageArtifactValidationError(
            "message_ids must be an iterable of positive integers"
        )
    try:
        candidates = iter(message_ids)
    except TypeError as exc:
        raise MessageArtifactValidationError(
            "message_ids must be an iterable of positive integers"
        ) from exc

    normalized: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        message_id = _required_message_id(candidate)
        if message_id in seen:
            continue
        seen.add(message_id)
        normalized.append(message_id)
        if len(normalized) > MAX_BATCH_MESSAGE_IDS:
            raise MessageArtifactValidationError(
                f"message_ids cannot contain more than {MAX_BATCH_MESSAGE_IDS} unique IDs"
            )
    return tuple(normalized)


def _required_artifact_type(artifact_type: object) -> str:
    if not isinstance(artifact_type, str) or artifact_type not in ALLOWED_ARTIFACT_TYPES:
        raise MessageArtifactValidationError(
            "artifact_type must be one of "
            + ", ".join(sorted(ALLOWED_ARTIFACT_TYPES))
        )
    return artifact_type


def _validate_json_value(value: Any, *, depth: int = 0) -> JSONValue:
    if depth > MAX_JSON_DEPTH:
        raise MessageArtifactValidationError(
            f"artifact JSON cannot exceed depth {MAX_JSON_DEPTH}"
        )
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > MAX_JSON_STRING_CHARS:
            raise MessageArtifactValidationError(
                f"artifact strings cannot exceed {MAX_JSON_STRING_CHARS} characters"
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MessageArtifactValidationError(
                "artifact JSON cannot contain NaN or infinite numbers"
            )
        return value
    if isinstance(value, list):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise MessageArtifactValidationError(
                f"artifact arrays cannot exceed {MAX_COLLECTION_ITEMS} items"
            )
        return [_validate_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise MessageArtifactValidationError(
                f"artifact objects cannot exceed {MAX_COLLECTION_ITEMS} keys"
            )
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > MAX_JSON_KEY_CHARS:
                raise MessageArtifactValidationError(
                    f"artifact object keys must be 1-{MAX_JSON_KEY_CHARS} character strings"
                )
            result[key] = _validate_json_value(item, depth=depth + 1)
        return result
    raise MessageArtifactValidationError(
        f"artifact JSON contains unsupported type: {type(value).__name__}"
    )


def _encode_payload(payload: object) -> str:
    if not isinstance(payload, Mapping):
        raise MessageArtifactValidationError("artifact payload must be a JSON object")
    validated = _validate_json_value(payload)
    assert isinstance(validated, dict)
    encoded = json.dumps(
        validated,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) > MAX_ARTIFACT_JSON_BYTES:
        raise MessageArtifactValidationError(
            f"artifact JSON cannot exceed {MAX_ARTIFACT_JSON_BYTES} bytes"
        )
    return encoded


def _decode_payload(encoded: object) -> dict[str, JSONValue]:
    if not isinstance(encoded, str):
        raise MessageArtifactSchemaError("artifact payload_json must be text")
    if len(encoded.encode("utf-8")) > MAX_ARTIFACT_JSON_BYTES:
        raise MessageArtifactSchemaError("stored artifact exceeds the JSON size limit")
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise MessageArtifactSchemaError("stored artifact contains invalid JSON") from exc
    try:
        validated = _validate_json_value(decoded)
    except MessageArtifactValidationError as exc:
        raise MessageArtifactSchemaError("stored artifact contains invalid JSON types") from exc
    if not isinstance(validated, dict):
        raise MessageArtifactSchemaError("stored artifact payload must be an object")
    return validated


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _connect(db_path: DatabasePath) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_MS / 1_000)
    try:
        init_message_artifacts_schema(conn)
        conn.commit()
    except Exception:
        conn.close()
        raise
    return conn


def save_message_artifact(
    db_path: DatabasePath,
    message_id: int,
    artifact_type: str,
    payload: Mapping[str, JSONValue],
) -> int:
    """Persist one validated artifact and return its immutable ID."""

    clean_message_id = _required_message_id(message_id)
    clean_artifact_type = _required_artifact_type(artifact_type)
    encoded = _encode_payload(payload)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if (
            conn.execute(
                "SELECT 1 FROM chat_history WHERE id = ?", (clean_message_id,)
            ).fetchone()
            is None
        ):
            raise MessageArtifactValidationError("message_id does not exist")
        cursor = conn.execute(
            f"INSERT INTO {_TABLE_NAME} "
            "(message_id, artifact_type, payload_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (clean_message_id, clean_artifact_type, encoded, _utc_now()),
        )
        artifact_id = int(cursor.lastrowid)
        conn.commit()
        return artifact_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_ui_artifacts(
    db_path: DatabasePath,
    message_id: int,
    artifacts: Iterable[UIArtifact],
) -> tuple[int, ...]:
    """Persist several UI artifacts atomically for one assistant message."""

    clean_message_id = _required_message_id(message_id)
    items = tuple(artifacts)
    if not all(isinstance(item, UIArtifact) for item in items):
        raise MessageArtifactValidationError("artifacts must contain UIArtifact values")
    prepared = [
        (
            _required_artifact_type(item.artifact_type),
            _encode_payload(item.payload),
        )
        for item in items
    ]
    if not prepared:
        return ()
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if (
            conn.execute(
                "SELECT 1 FROM chat_history WHERE id = ?", (clean_message_id,)
            ).fetchone()
            is None
        ):
            raise MessageArtifactValidationError("message_id does not exist")
        artifact_ids: list[int] = []
        for artifact_type, encoded in prepared:
            cursor = conn.execute(
                f"INSERT INTO {_TABLE_NAME} "
                "(message_id, artifact_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (clean_message_id, artifact_type, encoded, _utc_now()),
            )
            artifact_ids.append(int(cursor.lastrowid))
        conn.commit()
        return tuple(artifact_ids)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_message_artifacts(
    db_path: DatabasePath, message_id: int
) -> list[MessageArtifact]:
    clean_message_id = _required_message_id(message_id)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT id, message_id, artifact_type, payload_json, created_at "
            f"FROM {_TABLE_NAME} WHERE message_id = ? ORDER BY id ASC",
            (clean_message_id,),
        ).fetchall()
        return [
            MessageArtifact(
                id=int(row[0]),
                message_id=int(row[1]),
                artifact_type=str(row[2]),
                payload=_decode_payload(row[3]),
                created_at=str(row[4]),
            )
            for row in rows
        ]
    finally:
        conn.close()


def list_artifacts_for_messages(
    db_path: DatabasePath, message_ids: Iterable[int]
) -> dict[int, list[MessageArtifact]]:
    """Read artifacts for many messages with one bounded database query.

    Duplicate IDs are ignored while preserving first-seen key order.  Requested
    messages with no artifacts remain present with an empty list.
    """

    clean_message_ids = _normalize_message_ids(message_ids)
    if not clean_message_ids:
        return {}
    placeholders = ",".join("?" for _ in clean_message_ids)
    artifacts_by_message: dict[int, list[MessageArtifact]] = {
        message_id: [] for message_id in clean_message_ids
    }
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT id, message_id, artifact_type, payload_json, created_at "
            f"FROM {_TABLE_NAME} WHERE message_id IN ({placeholders}) "
            "ORDER BY message_id ASC, id ASC",
            clean_message_ids,
        ).fetchall()
        for row in rows:
            message_id = int(row[1])
            artifacts_by_message[message_id].append(
                MessageArtifact(
                    id=int(row[0]),
                    message_id=message_id,
                    artifact_type=str(row[2]),
                    payload=_decode_payload(row[3]),
                    created_at=str(row[4]),
                )
            )
        return artifacts_by_message
    finally:
        conn.close()


# Concise aliases for integration code.
save_artifact = save_message_artifact
list_artifacts = list_message_artifacts


__all__ = [
    "ALLOWED_ARTIFACT_TYPES",
    "MAX_BATCH_MESSAGE_IDS",
    "MAX_ARTIFACT_JSON_BYTES",
    "MessageArtifact",
    "MessageArtifactSchemaError",
    "MessageArtifactValidationError",
    "init_message_artifact_schema",
    "init_message_artifacts_schema",
    "list_artifacts",
    "list_artifacts_for_messages",
    "list_message_artifacts",
    "save_artifact",
    "save_message_artifact",
    "save_ui_artifacts",
]
