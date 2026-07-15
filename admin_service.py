"""Small, audited administration service for the local SQLite database.

The public read APIs return dataclasses and never expose password hashes.  Chat
content is omitted unless ``include_content=True``; content access then requires
an actor and is written to ``admin_audit_log`` in the same transaction.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TypeAlias

from account_auth import AccountAuthSchemaError, init_account_auth_schema


DatabasePath: TypeAlias = str | Path

BUSY_TIMEOUT_MS = 5_000
MAX_PAGE_SIZE = 500
MAX_OFFSET = 10_000_000
MAX_USERNAME_LENGTH = 256
MAX_SEARCH_LENGTH = 200
MAX_ACTION_LENGTH = 80
MAX_DETAIL_LENGTH = 4_000
MAX_PASSWORD_HASH_LENGTH = 1_024

TRANSLATION_STATUSES = (
    "validated",
    "fixed",
    "rejected",
    "failed",
    "none",
    "legacy_unverified",
)

_AUDIT_TABLE = "admin_audit_log"
_ACTION_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*")
_ARGON2_PREFIXES = ("$argon2id$", "$argon2i$", "$argon2d$")

_REQUIRED_APPLICATION_COLUMNS = {
    "users": {"username", "password_hash", "auth_version"},
    "chat_history": {
        "id",
        "username",
        "type",
        "content",
        "timestamp",
        "japanese_content",
        "audio_path",
        "translation_status",
        "translation_issue_code",
    },
    "user_memories": {
        "id",
        "username",
        "category",
        "memory_key",
        "memory_value",
        "source",
        "created_at",
        "updated_at",
    },
}

_EXPECTED_AUDIT_COLUMNS = (
    ("id", "INTEGER", 0, None, 1),
    ("actor", "TEXT", 1, None, 0),
    ("action", "TEXT", 1, None, 0),
    ("target_username", "TEXT", 0, None, 0),
    ("detail", "TEXT", 0, None, 0),
    ("created_at", "TEXT", 1, None, 0),
)

_CREATE_AUDIT_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_AUDIT_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT NOT NULL CHECK (length(actor) BETWEEN 1 AND {MAX_USERNAME_LENGTH}),
    action TEXT NOT NULL CHECK (length(action) BETWEEN 1 AND {MAX_ACTION_LENGTH}),
    target_username TEXT CHECK (
        target_username IS NULL
        OR length(target_username) BETWEEN 1 AND {MAX_USERNAME_LENGTH}
    ),
    detail TEXT CHECK (detail IS NULL OR length(detail) <= {MAX_DETAIL_LENGTH}),
    created_at TEXT NOT NULL
)
"""


class AdminServiceError(RuntimeError):
    """Base error for the administration service."""


class AdminValidationError(AdminServiceError, ValueError):
    """Raised when an argument violates the public service contract."""


class AdminSchemaError(AdminServiceError):
    """Raised when the application or audit schema is incompatible."""


class AdminNotFoundError(AdminServiceError, LookupError):
    """Raised when a requested user does not exist."""


@dataclass(frozen=True)
class OverviewStats:
    total_users: int
    total_messages: int
    human_messages: int
    ai_messages: int
    total_memories: int
    users_with_messages: int
    latest_message_at: str | None


@dataclass(frozen=True)
class TranslationBreakdown:
    validated: int
    fixed: int
    rejected: int
    failed: int
    none: int
    legacy_unverified: int
    total_ai_messages: int


@dataclass(frozen=True)
class UserSummary:
    username: str
    message_count: int
    memory_count: int
    last_message_at: str | None


@dataclass(frozen=True)
class AdminMessage:
    id: int
    username: str
    type: str
    timestamp: str
    translation_status: str
    translation_issue_code: str | None
    has_japanese: bool
    has_audio: bool
    content: str | None
    japanese_content: str | None


@dataclass(frozen=True)
class AuditEntry:
    id: int
    actor: str
    action: str
    target_username: str | None
    detail: str | None
    created_at: str


@dataclass(frozen=True)
class DeletedUserResult:
    username: str
    messages_deleted: int
    memories_deleted: int
    user_deleted: bool


@dataclass(frozen=True)
class DatabaseHealth:
    ok: bool
    integrity_check: str
    foreign_key_violations: int
    journal_mode: str
    user_version: int
    db_size_bytes: int
    schema_ok: bool
    schema_issue: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _database_path(db_path: DatabasePath) -> Path:
    if not isinstance(db_path, (str, Path)):
        raise AdminValidationError("db_path must be a string or Path")
    if isinstance(db_path, str) and not db_path.strip():
        raise AdminValidationError("db_path cannot be empty")
    path = Path(db_path).expanduser()
    if not path.exists() or not path.is_file():
        raise AdminValidationError("database file does not exist")
    return path


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
        raise AdminSchemaError("SQLite foreign_keys could not be enabled")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


def _column_rows(
    conn: sqlite3.Connection, table_name: str
) -> list[tuple[object, ...]]:
    if table_name not in {*_REQUIRED_APPLICATION_COLUMNS, _AUDIT_TABLE}:
        raise AdminSchemaError("unsupported table requested for schema inspection")
    return conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()


def _audit_schema_issue(conn: sqlite3.Connection) -> str | None:
    if not _table_exists(conn, _AUDIT_TABLE):
        return f"missing table: {_AUDIT_TABLE}"
    actual = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), row[4], int(row[5]))
        for row in _column_rows(conn, _AUDIT_TABLE)
    )
    if actual != _EXPECTED_AUDIT_COLUMNS:
        return f"incompatible {_AUDIT_TABLE} columns"
    return None


def _application_schema_issues(conn: sqlite3.Connection) -> list[str]:
    issues: list[str] = []
    for table_name, required_columns in _REQUIRED_APPLICATION_COLUMNS.items():
        if not _table_exists(conn, table_name):
            issues.append(f"missing table: {table_name}")
            continue
        rows = _column_rows(conn, table_name)
        columns = {str(row[1]) for row in rows}
        missing = sorted(required_columns - columns)
        if missing:
            issues.append(f"{table_name} missing columns: {', '.join(missing)}")
        if table_name in {"chat_history", "user_memories"}:
            id_row = next((row for row in rows if str(row[1]) == "id"), None)
            if id_row is not None and int(id_row[5]) != 1:
                issues.append(f"{table_name}.id must be the primary key")
        if table_name == "users":
            username_row = next(
                (row for row in rows if str(row[1]) == "username"), None
            )
            if username_row is not None and int(username_row[5]) != 1:
                issues.append("users.username must be the primary key")
    return issues


def init_admin_schema(conn: sqlite3.Connection) -> None:
    """Create and validate the additive audit table on a caller-owned connection.

    The function intentionally does not commit.
    """

    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be a sqlite3.Connection")
    _configure_connection(conn)
    try:
        init_account_auth_schema(conn)
    except AccountAuthSchemaError as exc:
        raise AdminSchemaError(str(exc)) from exc
    if not _table_exists(conn, _AUDIT_TABLE):
        conn.execute(_CREATE_AUDIT_TABLE_SQL)
    issue = _audit_schema_issue(conn)
    if issue is not None:
        raise AdminSchemaError(issue)


def _connect(db_path: DatabasePath) -> sqlite3.Connection:
    path = _database_path(db_path)
    conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1_000)
    conn.row_factory = sqlite3.Row
    try:
        init_admin_schema(conn)
        issues = _application_schema_issues(conn)
        if issues:
            raise AdminSchemaError("; ".join(issues))
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    return conn


@contextmanager
def _immediate_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _clean_username(value: str, field_name: str = "username") -> str:
    if not isinstance(value, str):
        raise AdminValidationError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise AdminValidationError(f"{field_name} cannot be empty")
    if len(cleaned) > MAX_USERNAME_LENGTH:
        raise AdminValidationError(
            f"{field_name} cannot exceed {MAX_USERNAME_LENGTH} characters"
        )
    return cleaned


def _clean_search(value: str) -> str:
    if not isinstance(value, str):
        raise AdminValidationError("search must be a string")
    cleaned = value.strip()
    if len(cleaned) > MAX_SEARCH_LENGTH:
        raise AdminValidationError(
            f"search cannot exceed {MAX_SEARCH_LENGTH} characters"
        )
    return cleaned


def _clean_action(value: str) -> str:
    if not isinstance(value, str):
        raise AdminValidationError("action must be a string")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > MAX_ACTION_LENGTH
        or _ACTION_PATTERN.fullmatch(cleaned) is None
    ):
        raise AdminValidationError(
            "action must start with a lowercase letter and contain only "
            "lowercase letters, digits, dots, underscores, or hyphens"
        )
    return cleaned


def _clean_detail(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AdminValidationError("detail must be a string or None")
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_DETAIL_LENGTH:
        raise AdminValidationError(
            f"detail cannot exceed {MAX_DETAIL_LENGTH} characters"
        )
    return cleaned


def _clean_limit(value: int, default_name: str = "limit") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdminValidationError(f"{default_name} must be an integer")
    if not 1 <= value <= MAX_PAGE_SIZE:
        raise AdminValidationError(
            f"{default_name} must be between 1 and {MAX_PAGE_SIZE}"
        )
    return value


def _clean_offset(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdminValidationError("offset must be an integer")
    if not 0 <= value <= MAX_OFFSET:
        raise AdminValidationError(f"offset must be between 0 and {MAX_OFFSET}")
    return value


def _clean_password_hash(value: str) -> str:
    if not isinstance(value, str):
        raise AdminValidationError("password_hash must be a string")
    if value != value.strip() or any(character.isspace() for character in value):
        raise AdminValidationError("password_hash cannot contain whitespace")
    if not value.startswith(_ARGON2_PREFIXES):
        raise AdminValidationError("password_hash must be an encoded Argon2 hash")
    if len(value) > MAX_PASSWORD_HASH_LENGTH:
        raise AdminValidationError(
            f"password_hash cannot exceed {MAX_PASSWORD_HASH_LENGTH} characters"
        )
    return value


def _require_user(conn: sqlite3.Connection, username: str) -> None:
    if conn.execute(
        "SELECT 1 FROM users WHERE username = ?", (username,)
    ).fetchone() is None:
        raise AdminNotFoundError("username does not exist")


def _insert_audit(
    conn: sqlite3.Connection,
    actor: str,
    action: str,
    target_username: str | None,
    detail: str | None,
) -> int:
    cursor = conn.execute(
        f"INSERT INTO {_AUDIT_TABLE} "
        "(actor, action, target_username, detail, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (actor, action, target_username, detail, _utc_now()),
    )
    return int(cursor.lastrowid)


def _json_detail(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def get_overview(db_path: DatabasePath) -> OverviewStats:
    conn = _connect(db_path)
    try:
        total_users = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
        message_row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN type = 'human' THEN 1 ELSE 0 END), 0) AS human, "
            "COALESCE(SUM(CASE WHEN type = 'ai' THEN 1 ELSE 0 END), 0) AS ai, "
            "COUNT(DISTINCT username) AS users_with_messages "
            "FROM chat_history"
        ).fetchone()
        total_memories = int(
            conn.execute("SELECT COUNT(*) FROM user_memories").fetchone()[0]
        )
        latest_row = conn.execute(
            "SELECT timestamp FROM chat_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return OverviewStats(
            total_users=total_users,
            total_messages=int(message_row["total"]),
            human_messages=int(message_row["human"]),
            ai_messages=int(message_row["ai"]),
            total_memories=total_memories,
            users_with_messages=int(message_row["users_with_messages"]),
            latest_message_at=(
                None if latest_row is None else str(latest_row["timestamp"])
            ),
        )
    finally:
        conn.close()


def get_translation_breakdown(db_path: DatabasePath) -> TranslationBreakdown:
    conn = _connect(db_path)
    try:
        counts = {status: 0 for status in TRANSLATION_STATUSES}
        for row in conn.execute(
            "SELECT translation_status, COUNT(*) AS count "
            "FROM chat_history WHERE type = 'ai' GROUP BY translation_status"
        ):
            status = str(row["translation_status"])
            if status not in counts:
                raise AdminSchemaError(
                    f"chat_history contains unsupported translation status: {status}"
                )
            counts[status] = int(row["count"])
        return TranslationBreakdown(
            validated=counts["validated"],
            fixed=counts["fixed"],
            rejected=counts["rejected"],
            failed=counts["failed"],
            none=counts["none"],
            legacy_unverified=counts["legacy_unverified"],
            total_ai_messages=sum(counts.values()),
        )
    finally:
        conn.close()


def list_user_summaries(
    db_path: DatabasePath,
    search: str = "",
    limit: int = 200,
    offset: int = 0,
) -> list[UserSummary]:
    clean_search = _clean_search(search)
    clean_limit = _clean_limit(limit)
    clean_offset = _clean_offset(offset)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "WITH message_stats AS ("
            "  SELECT username, COUNT(*) AS message_count, MAX(id) AS last_message_id "
            "  FROM chat_history GROUP BY username"
            "), memory_stats AS ("
            "  SELECT username, COUNT(*) AS memory_count "
            "  FROM user_memories GROUP BY username"
            ") "
            "SELECT u.username, COALESCE(ms.message_count, 0) AS message_count, "
            "COALESCE(mem.memory_count, 0) AS memory_count, "
            "latest.timestamp AS last_message_at, ms.last_message_id "
            "FROM users AS u "
            "LEFT JOIN message_stats AS ms ON ms.username = u.username "
            "LEFT JOIN memory_stats AS mem ON mem.username = u.username "
            "LEFT JOIN chat_history AS latest ON latest.id = ms.last_message_id "
            "WHERE (? = '' OR instr(lower(u.username), lower(?)) > 0) "
            "ORDER BY (ms.last_message_id IS NULL), ms.last_message_id DESC, "
            "u.username COLLATE NOCASE ASC, u.username ASC LIMIT ? OFFSET ?",
            (clean_search, clean_search, clean_limit, clean_offset),
        ).fetchall()
        return [
            UserSummary(
                username=str(row["username"]),
                message_count=int(row["message_count"]),
                memory_count=int(row["memory_count"]),
                last_message_at=(
                    None
                    if row["last_message_at"] is None
                    else str(row["last_message_at"])
                ),
            )
            for row in rows
        ]
    finally:
        conn.close()


def list_recent_messages(
    db_path: DatabasePath,
    username: str,
    limit: int = 50,
    offset: int = 0,
    include_content: bool = False,
    actor: str | None = None,
) -> list[AdminMessage]:
    clean_username = _clean_username(username)
    clean_limit = _clean_limit(limit)
    clean_offset = _clean_offset(offset)
    if not isinstance(include_content, bool):
        raise AdminValidationError("include_content must be a boolean")
    clean_actor = None
    if include_content:
        if actor is None:
            raise AdminValidationError(
                "actor is required when include_content is True"
            )
        clean_actor = _clean_username(actor, "actor")

    content_columns = (
        "content, japanese_content"
        if include_content
        else "NULL AS content, NULL AS japanese_content"
    )
    query = (
        "SELECT id, username, type, timestamp, translation_status, "
        "translation_issue_code, "
        "CASE WHEN japanese_content IS NOT NULL AND trim(japanese_content) <> '' "
        "THEN 1 ELSE 0 END AS has_japanese, "
        "CASE WHEN audio_path IS NOT NULL AND trim(audio_path) <> '' "
        "THEN 1 ELSE 0 END AS has_audio, "
        f"{content_columns} FROM chat_history "
        "WHERE username = ? ORDER BY id DESC LIMIT ? OFFSET ?"
    )

    conn = _connect(db_path)
    try:
        if include_content:
            with _immediate_transaction(conn):
                _require_user(conn, clean_username)
                rows = conn.execute(
                    query, (clean_username, clean_limit, clean_offset)
                ).fetchall()
                _insert_audit(
                    conn,
                    clean_actor or "",
                    "messages.content_viewed",
                    clean_username,
                    _json_detail(
                        {
                            "limit": clean_limit,
                            "offset": clean_offset,
                            "returned_count": len(rows),
                        }
                    ),
                )
        else:
            _require_user(conn, clean_username)
            rows = conn.execute(
                query, (clean_username, clean_limit, clean_offset)
            ).fetchall()

        return [
            AdminMessage(
                id=int(row["id"]),
                username=str(row["username"]),
                type=str(row["type"]),
                timestamp=str(row["timestamp"]),
                translation_status=str(row["translation_status"]),
                translation_issue_code=(
                    None
                    if row["translation_issue_code"] is None
                    else str(row["translation_issue_code"])
                ),
                has_japanese=bool(row["has_japanese"]),
                has_audio=bool(row["has_audio"]),
                content=None if row["content"] is None else str(row["content"]),
                japanese_content=(
                    None
                    if row["japanese_content"] is None
                    else str(row["japanese_content"])
                ),
            )
            for row in rows
        ]
    finally:
        conn.close()


def list_audit_entries(
    db_path: DatabasePath, limit: int = 200
) -> list[AuditEntry]:
    clean_limit = _clean_limit(limit)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT id, actor, action, target_username, detail, created_at "
            f"FROM {_AUDIT_TABLE} ORDER BY id DESC LIMIT ?",
            (clean_limit,),
        ).fetchall()
        return [
            AuditEntry(
                id=int(row["id"]),
                actor=str(row["actor"]),
                action=str(row["action"]),
                target_username=(
                    None
                    if row["target_username"] is None
                    else str(row["target_username"])
                ),
                detail=None if row["detail"] is None else str(row["detail"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]
    finally:
        conn.close()


def record_admin_action(
    db_path: DatabasePath,
    actor: str,
    action: str,
    target_username: str | None = None,
    detail: str | None = None,
) -> int:
    clean_actor = _clean_username(actor, "actor")
    clean_action = _clean_action(action)
    clean_target = (
        None
        if target_username is None
        else _clean_username(target_username, "target_username")
    )
    clean_detail = _clean_detail(detail)
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            return _insert_audit(
                conn, clean_actor, clean_action, clean_target, clean_detail
            )
    finally:
        conn.close()


def replace_user_password_hash(
    db_path: DatabasePath,
    actor: str,
    username: str,
    password_hash: str,
) -> bool:
    clean_actor = _clean_username(actor, "actor")
    clean_username = _clean_username(username)
    clean_hash = _clean_password_hash(password_hash)
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            _require_user(conn, clean_username)
            cursor = conn.execute(
                "UPDATE users SET password_hash = ?, auth_version = auth_version + 1 "
                "WHERE username = ?",
                (clean_hash, clean_username),
            )
            if cursor.rowcount != 1:
                raise AdminServiceError("password hash update affected an unexpected row count")
            _insert_audit(
                conn,
                clean_actor,
                "user.password_hash_replaced",
                clean_username,
                None,
            )
        return True
    finally:
        conn.close()


def clear_user_history(
    db_path: DatabasePath, actor: str, username: str
) -> int:
    clean_actor = _clean_username(actor, "actor")
    clean_username = _clean_username(username)
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            _require_user(conn, clean_username)
            cursor = conn.execute(
                "DELETE FROM chat_history WHERE username = ?", (clean_username,)
            )
            deleted = int(cursor.rowcount)
            _insert_audit(
                conn,
                clean_actor,
                "user.history_cleared",
                clean_username,
                _json_detail({"deleted_count": deleted}),
            )
        return deleted
    finally:
        conn.close()


def clear_user_memories(
    db_path: DatabasePath, actor: str, username: str
) -> int:
    clean_actor = _clean_username(actor, "actor")
    clean_username = _clean_username(username)
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            _require_user(conn, clean_username)
            cursor = conn.execute(
                "DELETE FROM user_memories WHERE username = ?", (clean_username,)
            )
            deleted = int(cursor.rowcount)
            _insert_audit(
                conn,
                clean_actor,
                "user.memories_cleared",
                clean_username,
                _json_detail({"deleted_count": deleted}),
            )
        return deleted
    finally:
        conn.close()


def delete_user_account(
    db_path: DatabasePath, actor: str, username: str
) -> DeletedUserResult:
    clean_actor = _clean_username(actor, "actor")
    clean_username = _clean_username(username)
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            _require_user(conn, clean_username)
            message_cursor = conn.execute(
                "DELETE FROM chat_history WHERE username = ?", (clean_username,)
            )
            memory_cursor = conn.execute(
                "DELETE FROM user_memories WHERE username = ?", (clean_username,)
            )
            user_cursor = conn.execute(
                "DELETE FROM users WHERE username = ?", (clean_username,)
            )
            if user_cursor.rowcount != 1:
                raise AdminServiceError("user deletion affected an unexpected row count")
            messages_deleted = int(message_cursor.rowcount)
            memories_deleted = int(memory_cursor.rowcount)
            _insert_audit(
                conn,
                clean_actor,
                "user.account_deleted",
                clean_username,
                _json_detail(
                    {
                        "memories_deleted": memories_deleted,
                        "messages_deleted": messages_deleted,
                    }
                ),
            )
        return DeletedUserResult(
            username=clean_username,
            messages_deleted=messages_deleted,
            memories_deleted=memories_deleted,
            user_deleted=True,
        )
    finally:
        conn.close()


def get_database_health(db_path: DatabasePath) -> DatabaseHealth:
    path = _database_path(db_path)
    conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1_000)
    try:
        _configure_connection(conn)
        schema_issues: list[str] = []
        try:
            init_admin_schema(conn)
            schema_issues.extend(_application_schema_issues(conn))
            if schema_issues:
                conn.rollback()
            else:
                conn.commit()
        except AdminSchemaError as exc:
            conn.rollback()
            schema_issues.append(str(exc))
            schema_issues.extend(_application_schema_issues(conn))

        integrity_rows = conn.execute("PRAGMA integrity_check").fetchall()
        integrity_check = "; ".join(str(row[0]) for row in integrity_rows)
        foreign_key_violations = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        schema_ok = not schema_issues
        return DatabaseHealth(
            ok=(
                integrity_check.lower() == "ok"
                and foreign_key_violations == 0
                and schema_ok
            ),
            integrity_check=integrity_check,
            foreign_key_violations=foreign_key_violations,
            journal_mode=journal_mode,
            user_version=user_version,
            db_size_bytes=path.stat().st_size,
            schema_ok=schema_ok,
            schema_issue=None if schema_ok else "; ".join(schema_issues),
        )
    finally:
        conn.close()
