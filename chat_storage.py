"""User-scoped, transactional SQLite storage for chat exchanges."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias


DatabasePath: TypeAlias = str | Path

BUSY_TIMEOUT_MS = 5_000
TRANSLATION_STATUSES = (
    "validated",
    "fixed",
    "rejected",
    "failed",
    "none",
    "legacy_unverified",
)
PLAYABLE_TRANSLATION_STATUSES = ("validated", "fixed")

_TABLE_NAME = "chat_history"
_SELECT_COLUMNS = (
    "id, username, type, content, japanese_content, audio_path, "
    "translation_status, translation_issue_code, timestamp"
)
_BASE_REQUIRED_COLUMNS = {
    "id",
    "username",
    "type",
    "content",
    "japanese_content",
    "audio_path",
    "timestamp",
}
_ADDITIVE_COLUMNS = {"translation_status", "translation_issue_code"}


class ChatStorageSchemaError(RuntimeError):
    """Raised when the existing users or chat-history schema is incompatible."""


class ChatStorageValidationError(ValueError):
    """Raised when a chat-storage request violates the public contract."""


@dataclass(frozen=True)
class StoredMessage:
    id: int
    username: str
    type: str
    content: str
    japanese_content: str | None
    audio_path: str | None
    translation_status: str
    translation_issue_code: str | None
    timestamp: str


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


def _column_rows(conn: sqlite3.Connection) -> dict[str, tuple[object, ...]]:
    return {
        str(row[1]): row
        for row in conn.execute(f"PRAGMA table_info({_TABLE_NAME})").fetchall()
    }


def _validate_translation_status_values(conn: sqlite3.Connection) -> None:
    placeholders = ", ".join("?" for _ in TRANSLATION_STATUSES)
    invalid_statuses = conn.execute(
        f"SELECT DISTINCT translation_status FROM {_TABLE_NAME} "
        f"WHERE translation_status IS NULL "
        f"OR translation_status NOT IN ({placeholders})",
        TRANSLATION_STATUSES,
    ).fetchall()
    if invalid_statuses:
        raise ChatStorageSchemaError(
            f"chat_history contains invalid translation statuses: {invalid_statuses!r}"
        )


def _validate_migration_prerequisites(
    conn: sqlite3.Connection,
) -> dict[str, tuple[object, ...]]:
    """Validate every invariant that the additive migration cannot repair."""

    if not _table_exists(conn, "users"):
        raise ChatStorageSchemaError("Required users table is missing")
    user_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    if "username" not in user_columns:
        raise ChatStorageSchemaError("users.username column is missing")
    if not _table_exists(conn, _TABLE_NAME):
        raise ChatStorageSchemaError("Required chat_history table is missing")

    columns = _column_rows(conn)
    missing = sorted(_BASE_REQUIRED_COLUMNS - set(columns))
    if missing:
        raise ChatStorageSchemaError(
            f"chat_history is missing required columns: {', '.join(missing)}"
        )
    if int(columns["id"][5]) != 1:
        raise ChatStorageSchemaError("chat_history.id must be the primary key")

    if "translation_status" in columns:
        _validate_translation_status_values(conn)
    return columns


def _validate_required_schema(conn: sqlite3.Connection) -> None:
    columns = _validate_migration_prerequisites(conn)
    missing = sorted(_ADDITIVE_COLUMNS - set(columns))
    if missing:
        raise ChatStorageSchemaError(
            f"chat_history is missing required columns: {', '.join(missing)}"
        )


def init_chat_storage_schema(conn: sqlite3.Connection) -> None:
    """Add translation audit fields to the existing chat-history table.

    The caller owns the connection and transaction. This function does not
    commit, so migration can participate in the caller's startup transaction.
    """

    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be a sqlite3.Connection")

    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
        raise ChatStorageSchemaError(
            "SQLite foreign_keys could not be enabled; initialize outside an active transaction"
        )
    # Reject incompatible schemas and bad existing status data before the first
    # ALTER, so a database that cannot be repaired is left completely untouched.
    original_columns = _validate_migration_prerequisites(conn)

    started_transaction = not conn.in_transaction
    savepoint_name = "chat_storage_schema_migration"
    savepoint_open = False
    try:
        if started_transaction:
            # Keep the transaction open for the caller, as promised by this
            # function's public contract. The savepoint makes DDL rollback-safe.
            conn.execute("BEGIN IMMEDIATE")
        conn.execute(f"SAVEPOINT {savepoint_name}")
        savepoint_open = True

        if "translation_status" not in original_columns:
            allowed_sql = ", ".join(f"'{status}'" for status in TRANSLATION_STATUSES)
            conn.execute(
                f"ALTER TABLE {_TABLE_NAME} ADD COLUMN translation_status "
                f"TEXT NOT NULL DEFAULT 'none' "
                f"CHECK (translation_status IN ({allowed_sql}))"
            )
        if "translation_issue_code" not in original_columns:
            conn.execute(
                f"ALTER TABLE {_TABLE_NAME} ADD COLUMN translation_issue_code TEXT"
            )

        conn.execute(
            f"UPDATE {_TABLE_NAME} SET translation_status = "
            "CASE "
            "WHEN type = 'ai' "
            "AND japanese_content IS NOT NULL "
            "AND trim(japanese_content) <> '' "
            "THEN 'legacy_unverified' "
            "ELSE 'none' END "
            "WHERE translation_status IS NULL "
            "OR (translation_status = 'none' AND type = 'ai' "
            "AND japanese_content IS NOT NULL AND trim(japanese_content) <> '')"
        )

        _validate_required_schema(conn)
        conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        savepoint_open = False
    except Exception:
        if savepoint_open:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            finally:
                conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        if started_transaction:
            conn.rollback()
        raise


def _connect(db_path: DatabasePath) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_MS / 1_000)
    try:
        init_chat_storage_schema(conn)
        conn.commit()
    except Exception:
        conn.close()
        raise
    return conn


def _required_username(username: str) -> str:
    if not isinstance(username, str):
        raise ChatStorageValidationError("username must be a string")
    cleaned = username.strip()
    if not cleaned:
        raise ChatStorageValidationError("username cannot be empty")
    return cleaned


def _required_message_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ChatStorageValidationError(f"{field_name} must be a string")
    if not value.strip():
        raise ChatStorageValidationError(f"{field_name} cannot be empty")
    return value


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ChatStorageValidationError(f"{field_name} must be a string or None")
    cleaned = value.strip()
    return cleaned or None


def _required_message_id(message_id: int) -> int:
    if isinstance(message_id, bool) or not isinstance(message_id, int) or message_id <= 0:
        raise ChatStorageValidationError("message_id must be a positive integer")
    return message_id


def _required_translation_status(status: str) -> str:
    if not isinstance(status, str) or status not in TRANSLATION_STATUSES:
        raise ChatStorageValidationError(
            f"translation_status must be one of {', '.join(TRANSLATION_STATUSES)}"
        )
    return status


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _row_to_message(row: tuple[object, ...]) -> StoredMessage:
    return StoredMessage(
        id=int(row[0]),
        username=str(row[1]),
        type=str(row[2]),
        content=str(row[3]),
        japanese_content=None if row[4] is None else str(row[4]),
        audio_path=None if row[5] is None else str(row[5]),
        translation_status=str(row[6]),
        translation_issue_code=None if row[7] is None else str(row[7]),
        timestamp=str(row[8]),
    )


def list_messages(db_path: DatabasePath, username: str) -> list[StoredMessage]:
    """Return one user's messages in immutable insertion-ID order."""

    cleaned_username = _required_username(username)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM {_TABLE_NAME} "
            "WHERE username = ? ORDER BY id ASC",
            (cleaned_username,),
        ).fetchall()
        return [_row_to_message(row) for row in rows]
    finally:
        conn.close()


def save_exchange(
    db_path: DatabasePath,
    username: str,
    user_text: str,
    ai_text: str,
    japanese_text: str | None,
    translation_status: str,
    translation_issue_code: str | None,
    audio_path: str | None,
) -> tuple[int, int]:
    """Atomically persist the human and AI sides of one exchange."""

    cleaned_username = _required_username(username)
    clean_user_text = _required_message_text(user_text, "user_text")
    clean_ai_text = _required_message_text(ai_text, "ai_text")
    clean_status = _required_translation_status(translation_status)
    clean_japanese = _optional_text(japanese_text, "japanese_text")
    clean_issue = _optional_text(translation_issue_code, "translation_issue_code")
    clean_audio = _optional_text(audio_path, "audio_path")

    translation_is_playable = clean_status in PLAYABLE_TRANSLATION_STATUSES
    if translation_is_playable and clean_japanese is None:
        raise ChatStorageValidationError(
            "validated or fixed translations require japanese_text"
        )
    stored_japanese = clean_japanese if translation_is_playable else None
    stored_audio = clean_audio if translation_is_playable else None

    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        user_exists = conn.execute(
            "SELECT 1 FROM users WHERE username = ?",
            (cleaned_username,),
        ).fetchone()
        if user_exists is None:
            raise ChatStorageValidationError("username does not exist")

        timestamp = _utc_now()
        user_cursor = conn.execute(
            f"INSERT INTO {_TABLE_NAME} "
            "(username, type, content, japanese_content, audio_path, "
            "translation_status, translation_issue_code, timestamp) "
            "VALUES (?, 'human', ?, NULL, NULL, 'none', NULL, ?)",
            (cleaned_username, clean_user_text, timestamp),
        )
        ai_cursor = conn.execute(
            f"INSERT INTO {_TABLE_NAME} "
            "(username, type, content, japanese_content, audio_path, "
            "translation_status, translation_issue_code, timestamp) "
            "VALUES (?, 'ai', ?, ?, ?, ?, ?, ?)",
            (
                cleaned_username,
                clean_ai_text,
                stored_japanese,
                stored_audio,
                clean_status,
                clean_issue,
                timestamp,
            ),
        )
        user_id = int(user_cursor.lastrowid)
        ai_id = int(ai_cursor.lastrowid)
        conn.commit()
        return user_id, ai_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_message_audio(
    db_path: DatabasePath,
    username: str,
    message_id: int,
    audio_path: str,
) -> bool:
    """Attach audio only to the user's AI message with an approved translation."""

    cleaned_username = _required_username(username)
    clean_message_id = _required_message_id(message_id)
    clean_audio = _optional_text(audio_path, "audio_path")
    if clean_audio is None:
        raise ChatStorageValidationError("audio_path cannot be empty")

    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            f"UPDATE {_TABLE_NAME} SET audio_path = ? "
            "WHERE id = ? AND username = ? AND type = 'ai' "
            "AND translation_status IN ('validated', 'fixed')",
            (clean_audio, clean_message_id, cleaned_username),
        )
        conn.commit()
        return cursor.rowcount == 1
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
