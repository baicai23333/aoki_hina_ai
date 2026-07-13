"""Explicit, user-scoped structured memory storage for the Streamlit app.

This module intentionally does not infer memories from chat history.  Callers
must only persist values that a signed-in user explicitly chose to save.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias


DatabasePath: TypeAlias = str | Path

MEMORY_CATEGORIES = (
    "preferred_name",
    "interest",
    "goal",
    "conversation_preference",
)
MAX_MEMORIES_PER_USER = 50
MAX_MEMORY_KEY_LENGTH = 80
MAX_MEMORY_VALUE_LENGTH = 500
BUSY_TIMEOUT_MS = 5_000
MEMORY_SOURCE = "manual_ui"

_TABLE_NAME = "user_memories"
_ORDER_INDEX_NAME = "idx_user_memories_username_updated"
_SELECT_COLUMNS = (
    "id, username, category, memory_key, memory_value, source, "
    "created_at, updated_at"
)


class UserMemorySchemaError(RuntimeError):
    """Raised when an existing memory table is incompatible with this module."""


class UserMemoryValidationError(ValueError):
    """Raised when a memory request violates the public storage contract."""


class UserMemoryLimitError(UserMemoryValidationError):
    """Raised when a user tries to create more than the allowed memories."""


@dataclass(frozen=True)
class UserMemory:
    id: int
    username: str
    category: str
    memory_key: str
    memory_value: str
    source: str
    created_at: str
    updated_at: str


_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    category TEXT NOT NULL CHECK (
        category IN ('preferred_name', 'interest', 'goal', 'conversation_preference')
    ),
    memory_key TEXT NOT NULL CHECK (length(memory_key) BETWEEN 1 AND 80),
    memory_value TEXT NOT NULL CHECK (length(memory_value) BETWEEN 1 AND 500),
    source TEXT NOT NULL DEFAULT 'manual_ui' CHECK (source = 'manual_ui'),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE,
    UNIQUE (username, category, memory_key)
)
"""

_CREATE_ORDER_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS {_ORDER_INDEX_NAME}
ON {_TABLE_NAME}(username, updated_at DESC)
"""

_EXPECTED_COLUMNS = (
    ("id", "INTEGER", 0, None, 1),
    ("username", "TEXT", 1, None, 0),
    ("category", "TEXT", 1, None, 0),
    ("memory_key", "TEXT", 1, None, 0),
    ("memory_value", "TEXT", 1, None, 0),
    ("source", "TEXT", 1, "'manual_ui'", 0),
    ("created_at", "TEXT", 1, None, 0),
    ("updated_at", "TEXT", 1, None, 0),
)

_REQUIRED_TABLE_SQL_FRAGMENTS = (
    "idintegerprimarykeyautoincrement",
    "categorytextnotnullcheck(categoryin('preferred_name','interest','goal','conversation_preference'))",
    "memory_keytextnotnullcheck(length(memory_key)between1and80)",
    "memory_valuetextnotnullcheck(length(memory_value)between1and500)",
    "sourcetextnotnulldefault'manual_ui'check(source='manual_ui')",
    "foreignkey(username)referencesusers(username)ondeletecascade",
    "unique(username,category,memory_key)",
)


def _compact_schema_sql(value: str) -> str:
    compact = re.sub(r"\s+", "", value.lower())
    return compact.translate(str.maketrans("", "", '"`[]'))


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _validate_table_schema(conn: sqlite3.Connection) -> None:
    rows = conn.execute(f"PRAGMA table_info({_TABLE_NAME})").fetchall()
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), row[4], int(row[5]))
        for row in rows
    )
    if actual_columns != _EXPECTED_COLUMNS:
        raise UserMemorySchemaError(
            f"Incompatible {_TABLE_NAME} columns: expected {_EXPECTED_COLUMNS!r}, "
            f"got {actual_columns!r}"
        )

    schema_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (_TABLE_NAME,),
    ).fetchone()
    if schema_row is None or not schema_row[0]:
        raise UserMemorySchemaError(f"Missing CREATE TABLE SQL for {_TABLE_NAME}")
    compact_sql = _compact_schema_sql(str(schema_row[0]))
    missing_fragments = [
        fragment for fragment in _REQUIRED_TABLE_SQL_FRAGMENTS if fragment not in compact_sql
    ]
    if missing_fragments:
        raise UserMemorySchemaError(
            f"Incompatible {_TABLE_NAME} constraints: missing {missing_fragments!r}"
        )

    foreign_keys = conn.execute(f"PRAGMA foreign_key_list({_TABLE_NAME})").fetchall()
    expected_foreign_key = (
        "users",
        "username",
        "username",
        "NO ACTION",
        "CASCADE",
        "NONE",
    )
    actual_foreign_keys = {
        (
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[5]).upper(),
            str(row[6]).upper(),
            str(row[7]).upper(),
        )
        for row in foreign_keys
    }
    if actual_foreign_keys != {expected_foreign_key}:
        raise UserMemorySchemaError(
            f"Incompatible {_TABLE_NAME} foreign keys: {actual_foreign_keys!r}"
        )

    unique_key_found = False
    for index_row in conn.execute(f"PRAGMA index_list({_TABLE_NAME})").fetchall():
        if not int(index_row[2]):
            continue
        index_name = str(index_row[1])
        columns = tuple(
            str(row[2])
            for row in conn.execute(
                f"PRAGMA index_info({_quoted_identifier(index_name)})"
            ).fetchall()
        )
        if columns == ("username", "category", "memory_key"):
            unique_key_found = True
            break
    if not unique_key_found:
        raise UserMemorySchemaError(
            f"{_TABLE_NAME} is missing the unique username/category/memory_key constraint"
        )


def _validate_order_index(conn: sqlite3.Connection) -> None:
    index_rows = conn.execute(f"PRAGMA index_list({_TABLE_NAME})").fetchall()
    matching = [row for row in index_rows if str(row[1]) == _ORDER_INDEX_NAME]
    if len(matching) != 1 or int(matching[0][2]) != 0:
        raise UserMemorySchemaError(f"Incompatible {_ORDER_INDEX_NAME} index definition")

    key_columns = tuple(
        (str(row[2]), int(row[3]))
        for row in conn.execute(
            f"PRAGMA index_xinfo({_quoted_identifier(_ORDER_INDEX_NAME)})"
        ).fetchall()
        if int(row[5]) == 1 and row[2] is not None
    )
    if key_columns != (("username", 0), ("updated_at", 1)):
        raise UserMemorySchemaError(
            f"Incompatible {_ORDER_INDEX_NAME} columns: {key_columns!r}"
        )


def init_user_memory_schema(conn: sqlite3.Connection) -> None:
    """Create and validate the additive memory schema on an existing connection.

    The caller owns the connection and transaction.  This function deliberately
    does not commit, so it cannot accidentally commit unrelated caller work.
    """

    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be a sqlite3.Connection")

    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    foreign_keys_enabled = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    if foreign_keys_enabled != 1:
        raise UserMemorySchemaError(
            "SQLite foreign_keys could not be enabled; initialize outside an active transaction"
        )

    conn.execute(_CREATE_TABLE_SQL)
    _validate_table_schema(conn)
    conn.execute(_CREATE_ORDER_INDEX_SQL)
    _validate_order_index(conn)


def _connect(db_path: DatabasePath) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_MS / 1_000)
    try:
        init_user_memory_schema(conn)
        conn.commit()
    except Exception:
        conn.close()
        raise
    return conn


def _clean_username(username: str) -> str:
    if not isinstance(username, str):
        raise UserMemoryValidationError("username must be a string")
    cleaned = username.strip()
    if not cleaned:
        raise UserMemoryValidationError("username cannot be empty")
    return cleaned


def _clean_memory_fields(
    category: str,
    memory_key: str,
    memory_value: str,
) -> tuple[str, str, str]:
    if not isinstance(category, str):
        raise UserMemoryValidationError("category must be a string")
    cleaned_category = category.strip()
    if cleaned_category not in MEMORY_CATEGORIES:
        raise UserMemoryValidationError(
            f"category must be one of {', '.join(MEMORY_CATEGORIES)}"
        )

    if not isinstance(memory_key, str):
        raise UserMemoryValidationError("memory_key must be a string")
    cleaned_key = memory_key.strip()
    if not 1 <= len(cleaned_key) <= MAX_MEMORY_KEY_LENGTH:
        raise UserMemoryValidationError(
            f"memory_key length must be 1..{MAX_MEMORY_KEY_LENGTH}"
        )

    if not isinstance(memory_value, str):
        raise UserMemoryValidationError("memory_value must be a string")
    cleaned_value = memory_value.strip()
    if not 1 <= len(cleaned_value) <= MAX_MEMORY_VALUE_LENGTH:
        raise UserMemoryValidationError(
            f"memory_value length must be 1..{MAX_MEMORY_VALUE_LENGTH}"
        )
    return cleaned_category, cleaned_key, cleaned_value


def _clean_memory_id(memory_id: int) -> int:
    if isinstance(memory_id, bool) or not isinstance(memory_id, int) or memory_id <= 0:
        raise UserMemoryValidationError("memory_id must be a positive integer")
    return memory_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _row_to_memory(row: tuple[object, ...]) -> UserMemory:
    return UserMemory(
        id=int(row[0]),
        username=str(row[1]),
        category=str(row[2]),
        memory_key=str(row[3]),
        memory_value=str(row[4]),
        source=str(row[5]),
        created_at=str(row[6]),
        updated_at=str(row[7]),
    )


def list_memories(db_path: DatabasePath, username: str) -> list[UserMemory]:
    """Return at most the configured maximum, scoped to exactly one user."""

    cleaned_username = _clean_username(username)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM {_TABLE_NAME} "
            "WHERE username = ? ORDER BY updated_at DESC, id DESC LIMIT ?",
            (cleaned_username, MAX_MEMORIES_PER_USER),
        ).fetchall()
        return [_row_to_memory(row) for row in rows]
    finally:
        conn.close()


def upsert_memory(
    db_path: DatabasePath,
    username: str,
    category: str,
    memory_key: str,
    memory_value: str,
) -> UserMemory:
    """Create or replace one explicit manual memory for a signed-in user."""

    cleaned_username = _clean_username(username)
    cleaned_category, cleaned_key, cleaned_value = _clean_memory_fields(
        category, memory_key, memory_value
    )
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            f"SELECT id, created_at FROM {_TABLE_NAME} "
            "WHERE username = ? AND category = ? AND memory_key = ?",
            (cleaned_username, cleaned_category, cleaned_key),
        ).fetchone()
        now = _utc_now()
        if existing is None:
            count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {_TABLE_NAME} WHERE username = ?",
                    (cleaned_username,),
                ).fetchone()[0]
            )
            if count >= MAX_MEMORIES_PER_USER:
                raise UserMemoryLimitError(
                    f"a user can store at most {MAX_MEMORIES_PER_USER} memories"
                )
            cursor = conn.execute(
                f"INSERT INTO {_TABLE_NAME} "
                "(username, category, memory_key, memory_value, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cleaned_username,
                    cleaned_category,
                    cleaned_key,
                    cleaned_value,
                    MEMORY_SOURCE,
                    now,
                    now,
                ),
            )
            memory_id = int(cursor.lastrowid)
        else:
            memory_id = int(existing[0])
            conn.execute(
                f"UPDATE {_TABLE_NAME} "
                "SET memory_value = ?, source = ?, updated_at = ? "
                "WHERE id = ? AND username = ?",
                (cleaned_value, MEMORY_SOURCE, now, memory_id, cleaned_username),
            )

        row = conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM {_TABLE_NAME} WHERE id = ? AND username = ?",
            (memory_id, cleaned_username),
        ).fetchone()
        if row is None:
            raise RuntimeError("memory disappeared during upsert")
        conn.commit()
        return _row_to_memory(row)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        if "FOREIGN KEY constraint failed" in str(exc):
            raise UserMemoryValidationError("username does not exist") from exc
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_memory(db_path: DatabasePath, username: str, memory_id: int) -> bool:
    """Hard-delete one record, only when it belongs to the supplied username."""

    cleaned_username = _clean_username(username)
    cleaned_memory_id = _clean_memory_id(memory_id)
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            f"DELETE FROM {_TABLE_NAME} WHERE id = ? AND username = ?",
            (cleaned_memory_id, cleaned_username),
        )
        conn.commit()
        return cursor.rowcount == 1
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def clear_memories(db_path: DatabasePath, username: str) -> int:
    """Hard-delete every structured memory belonging to one user."""

    cleaned_username = _clean_username(username)
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            f"DELETE FROM {_TABLE_NAME} WHERE username = ?",
            (cleaned_username,),
        )
        conn.commit()
        return int(cursor.rowcount)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
