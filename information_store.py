"""Transactional SQLite storage for official, time-sensitive information.

The module owns four additive tables used by the background collector.  Schema
initialization is idempotent and deliberately fails before making changes when
an existing table is incompatible.  Public write APIs validate all untrusted
values and keep optional admin-audit writes in the same transaction.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence, TypeAlias
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


DatabasePath: TypeAlias = str | Path

BUSY_TIMEOUT_MS = 5_000
SOURCE_TYPES = ("html", "rss")
UPDATE_CATEGORIES = (
    "live",
    "event",
    "release",
    "broadcast",
    "goods",
    "ticket",
    "appearance",
    "announcement",
    "correction",
    "cancellation",
)
UPDATE_STATUSES = ("scheduled", "updated", "postponed", "cancelled", "completed")
VERIFICATION_STATUSES = ("pending", "approved", "rejected")
COLLECTOR_RUN_STATUSES = ("running", "succeeded", "partial", "failed")

_TABLES = (
    "information_sources",
    "source_documents",
    "official_updates",
    "collector_runs",
)
_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class InformationStoreError(RuntimeError):
    """Base error for the official-information store."""


class InformationSchemaError(InformationStoreError):
    """Raised when an existing database schema is incompatible."""


class InformationValidationError(InformationStoreError, ValueError):
    """Raised when a caller supplies invalid data."""


class InformationNotFoundError(InformationStoreError, LookupError):
    """Raised when a requested source, document, update, or run is absent."""


@dataclass(frozen=True)
class InformationSource:
    id: int
    name: str
    source_type: str
    base_url: str
    allowed_domains: tuple[str, ...]
    trust_level: int
    fetch_interval_minutes: int
    enabled: bool
    last_checked_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SourceDocument:
    id: int
    source_id: int
    canonical_url: str
    title: str | None
    raw_content: str
    content_hash: str
    published_at: str | None
    fetched_at: str
    last_modified_at: str | None


@dataclass(frozen=True)
class OfficialUpdate:
    id: int
    document_id: int
    source_id: int
    source_name: str
    canonical_url: str
    published_at: str | None
    category: str
    title: str
    summary: str | None
    event_start_at: str | None
    event_end_at: str | None
    venue: str | None
    status: str
    verification_status: str
    confidence: float
    replaces_update_id: int | None
    reviewed_by: str | None
    reviewed_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CollectorRun:
    id: int
    source_id: int
    started_at: str
    finished_at: str | None
    status: str
    discovered_count: int
    fetched_count: int
    new_document_count: int
    pending_update_count: int
    error_code: str | None
    detail: str | None
    source_name: str | None = None


_CREATE_TABLE_SQL = {
    "information_sources": """
        CREATE TABLE IF NOT EXISTS information_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 200),
            source_type TEXT NOT NULL CHECK (source_type IN ('html', 'rss')),
            base_url TEXT NOT NULL UNIQUE,
            allowed_domains_json TEXT NOT NULL,
            trust_level INTEGER NOT NULL CHECK (trust_level BETWEEN 0 AND 100),
            fetch_interval_minutes INTEGER NOT NULL CHECK (fetch_interval_minutes >= 1),
            enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
            last_checked_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """,
    "source_documents": """
        CREATE TABLE IF NOT EXISTS source_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            canonical_url TEXT NOT NULL,
            title TEXT,
            raw_content TEXT NOT NULL CHECK (length(raw_content) >= 1),
            content_hash TEXT NOT NULL CHECK (length(content_hash) = 64),
            published_at TEXT,
            fetched_at TEXT NOT NULL,
            last_modified_at TEXT,
            FOREIGN KEY (source_id) REFERENCES information_sources(id) ON DELETE RESTRICT,
            UNIQUE (source_id, content_hash)
        )
    """,
    "official_updates": """
        CREATE TABLE IF NOT EXISTS official_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL UNIQUE,
            category TEXT NOT NULL CHECK (category IN (
                'live', 'event', 'release', 'broadcast', 'goods', 'ticket',
                'appearance', 'announcement', 'correction', 'cancellation'
            )),
            title TEXT NOT NULL CHECK (length(title) BETWEEN 1 AND 500),
            summary TEXT,
            event_start_at TEXT,
            event_end_at TEXT,
            venue TEXT,
            status TEXT NOT NULL DEFAULT 'scheduled' CHECK (status IN (
                'scheduled', 'updated', 'postponed', 'cancelled', 'completed'
            )),
            verification_status TEXT NOT NULL DEFAULT 'pending' CHECK (
                verification_status IN ('pending', 'approved', 'rejected')
            ),
            confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
            replaces_update_id INTEGER,
            reviewed_by TEXT,
            reviewed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES source_documents(id) ON DELETE RESTRICT,
            FOREIGN KEY (replaces_update_id) REFERENCES official_updates(id) ON DELETE SET NULL
        )
    """,
    "collector_runs": """
        CREATE TABLE IF NOT EXISTS collector_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL CHECK (status IN (
                'running', 'succeeded', 'partial', 'failed'
            )),
            discovered_count INTEGER NOT NULL DEFAULT 0 CHECK (discovered_count >= 0),
            fetched_count INTEGER NOT NULL DEFAULT 0 CHECK (fetched_count >= 0),
            new_document_count INTEGER NOT NULL DEFAULT 0 CHECK (new_document_count >= 0),
            pending_update_count INTEGER NOT NULL DEFAULT 0 CHECK (pending_update_count >= 0),
            error_code TEXT,
            detail TEXT,
            FOREIGN KEY (source_id) REFERENCES information_sources(id) ON DELETE RESTRICT
        )
    """,
}

_EXPECTED_COLUMNS = {
    "information_sources": (
        ("id", "INTEGER", 0, None, 1),
        ("name", "TEXT", 1, None, 0),
        ("source_type", "TEXT", 1, None, 0),
        ("base_url", "TEXT", 1, None, 0),
        ("allowed_domains_json", "TEXT", 1, None, 0),
        ("trust_level", "INTEGER", 1, None, 0),
        ("fetch_interval_minutes", "INTEGER", 1, None, 0),
        ("enabled", "INTEGER", 1, "1", 0),
        ("last_checked_at", "TEXT", 0, None, 0),
        ("created_at", "TEXT", 1, None, 0),
        ("updated_at", "TEXT", 1, None, 0),
    ),
    "source_documents": (
        ("id", "INTEGER", 0, None, 1),
        ("source_id", "INTEGER", 1, None, 0),
        ("canonical_url", "TEXT", 1, None, 0),
        ("title", "TEXT", 0, None, 0),
        ("raw_content", "TEXT", 1, None, 0),
        ("content_hash", "TEXT", 1, None, 0),
        ("published_at", "TEXT", 0, None, 0),
        ("fetched_at", "TEXT", 1, None, 0),
        ("last_modified_at", "TEXT", 0, None, 0),
    ),
    "official_updates": (
        ("id", "INTEGER", 0, None, 1),
        ("document_id", "INTEGER", 1, None, 0),
        ("category", "TEXT", 1, None, 0),
        ("title", "TEXT", 1, None, 0),
        ("summary", "TEXT", 0, None, 0),
        ("event_start_at", "TEXT", 0, None, 0),
        ("event_end_at", "TEXT", 0, None, 0),
        ("venue", "TEXT", 0, None, 0),
        ("status", "TEXT", 1, "'scheduled'", 0),
        ("verification_status", "TEXT", 1, "'pending'", 0),
        ("confidence", "REAL", 1, None, 0),
        ("replaces_update_id", "INTEGER", 0, None, 0),
        ("reviewed_by", "TEXT", 0, None, 0),
        ("reviewed_at", "TEXT", 0, None, 0),
        ("created_at", "TEXT", 1, None, 0),
        ("updated_at", "TEXT", 1, None, 0),
    ),
    "collector_runs": (
        ("id", "INTEGER", 0, None, 1),
        ("source_id", "INTEGER", 1, None, 0),
        ("started_at", "TEXT", 1, None, 0),
        ("finished_at", "TEXT", 0, None, 0),
        ("status", "TEXT", 1, None, 0),
        ("discovered_count", "INTEGER", 1, "0", 0),
        ("fetched_count", "INTEGER", 1, "0", 0),
        ("new_document_count", "INTEGER", 1, "0", 0),
        ("pending_update_count", "INTEGER", 1, "0", 0),
        ("error_code", "TEXT", 0, None, 0),
        ("detail", "TEXT", 0, None, 0),
    ),
}

_REQUIRED_SQL_FRAGMENTS = {
    "information_sources": (
        "source_typetextnotnullcheck(source_typein('html','rss'))",
        "base_urltextnotnullunique",
        "check(enabledin(0,1))",
    ),
    "source_documents": (
        "foreignkey(source_id)referencesinformation_sources(id)ondeleterestrict",
        "unique(source_id,content_hash)",
    ),
    "official_updates": (
        "document_idintegernotnullunique",
        "verification_statusin('pending','approved','rejected')",
        "foreignkey(document_id)referencessource_documents(id)ondeleterestrict",
        "foreignkey(replaces_update_id)referencesofficial_updates(id)ondeletesetnull",
    ),
    "collector_runs": (
        "statusin('running','succeeded','partial','failed')",
        "foreignkey(source_id)referencesinformation_sources(id)ondeleterestrict",
    ),
}

_INDEX_SQL = {
    "idx_information_sources_enabled": (
        "information_sources",
        "CREATE INDEX IF NOT EXISTS idx_information_sources_enabled "
        "ON information_sources(enabled, last_checked_at)",
        ("enabled", "last_checked_at"),
    ),
    "idx_source_documents_canonical": (
        "source_documents",
        "CREATE INDEX IF NOT EXISTS idx_source_documents_canonical "
        "ON source_documents(source_id, canonical_url, fetched_at DESC)",
        ("source_id", "canonical_url", "fetched_at"),
    ),
    "idx_official_updates_recent": (
        "official_updates",
        "CREATE INDEX IF NOT EXISTS idx_official_updates_recent "
        "ON official_updates(verification_status, event_start_at, created_at)",
        ("verification_status", "event_start_at", "created_at"),
    ),
    "idx_collector_runs_source": (
        "collector_runs",
        "CREATE INDEX IF NOT EXISTS idx_collector_runs_source "
        "ON collector_runs(source_id, started_at DESC)",
        ("source_id", "started_at"),
    ),
}

_OFFICIAL_SELECT = (
    "SELECT u.id, u.document_id, d.source_id, s.name AS source_name, "
    "d.canonical_url, d.published_at, u.category, u.title, u.summary, "
    "u.event_start_at, u.event_end_at, u.venue, u.status, "
    "u.verification_status, u.confidence, u.replaces_update_id, "
    "u.reviewed_by, u.reviewed_at, u.created_at, u.updated_at "
    "FROM official_updates AS u "
    "JOIN source_documents AS d ON d.id = u.document_id "
    "JOIN information_sources AS s ON s.id = d.source_id "
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _compact_sql(value: str) -> str:
    return re.sub(r"[\s\"`\[\]]+", "", value.lower())


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone() is not None


def _validate_table(conn: sqlite3.Connection, table_name: str) -> None:
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    actual = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), row[4], int(row[5]))
        for row in rows
    )
    expected = _EXPECTED_COLUMNS[table_name]
    if actual != expected:
        raise InformationSchemaError(
            f"Incompatible {table_name} columns: expected {expected!r}, got {actual!r}"
        )
    sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if sql_row is None or not sql_row[0]:
        raise InformationSchemaError(f"Missing CREATE TABLE SQL for {table_name}")
    compact = _compact_sql(str(sql_row[0]))
    missing = [
        fragment
        for fragment in _REQUIRED_SQL_FRAGMENTS[table_name]
        if fragment not in compact
    ]
    if missing:
        raise InformationSchemaError(
            f"Incompatible {table_name} constraints: missing {missing!r}"
        )


def _validate_index(conn: sqlite3.Connection, name: str) -> None:
    table_name, _sql, expected_columns = _INDEX_SQL[name]
    rows = [
        row
        for row in conn.execute(f'PRAGMA index_list("{table_name}")').fetchall()
        if str(row[1]) == name
    ]
    if len(rows) != 1 or int(rows[0][2]) != 0:
        raise InformationSchemaError(f"Incompatible {name} index definition")
    columns = tuple(
        str(row[2])
        for row in conn.execute(f'PRAGMA index_info("{name}")').fetchall()
    )
    if columns != expected_columns:
        raise InformationSchemaError(
            f"Incompatible {name} columns: expected {expected_columns!r}, got {columns!r}"
        )


def _index_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        (name,),
    ).fetchone() is not None


def init_information_schema(conn: sqlite3.Connection) -> None:
    """Create and validate the additive information schema.

    The caller owns the connection and transaction.  Existing incompatible
    tables are rejected before any missing table or index is created.
    """

    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be a sqlite3.Connection")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
        raise InformationSchemaError(
            "SQLite foreign_keys could not be enabled; initialize outside an active transaction"
        )

    for table_name in _TABLES:
        if _table_exists(conn, table_name):
            _validate_table(conn, table_name)

    if all(_table_exists(conn, table_name) for table_name in _TABLES) and all(
        _index_exists(conn, name) for name in _INDEX_SQL
    ):
        for name in _INDEX_SQL:
            _validate_index(conn, name)
        return

    started_transaction = not conn.in_transaction
    savepoint = "information_schema_migration"
    savepoint_open = False
    try:
        if started_transaction:
            conn.execute("BEGIN IMMEDIATE")
        conn.execute(f"SAVEPOINT {savepoint}")
        savepoint_open = True
        for table_name in _TABLES:
            conn.execute(_CREATE_TABLE_SQL[table_name])
            _validate_table(conn, table_name)
        for name, (_table, sql, _columns) in _INDEX_SQL.items():
            conn.execute(sql)
            _validate_index(conn, name)
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        savepoint_open = False
    except Exception:
        if savepoint_open:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            finally:
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        if started_transaction:
            conn.rollback()
        raise


def _connect(db_path: DatabasePath) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_MS / 1_000)
    conn.row_factory = sqlite3.Row
    try:
        init_information_schema(conn)
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


def canonicalize_url(value: str) -> str:
    if not isinstance(value, str):
        raise InformationValidationError("URL must be a string")
    raw = value.strip()
    if not raw or len(raw) > 2_048:
        raise InformationValidationError("URL length must be 1..2048")
    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise InformationValidationError("URL must be absolute HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise InformationValidationError("URL credentials are not allowed")
    try:
        hostname = parsed.hostname.encode("idna").decode("ascii").lower()
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise InformationValidationError("URL host or port is invalid") from exc
    host_display = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host_display = f"{host_display}:{port}"
    filtered_query = []
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in _TRACKING_QUERY_KEYS:
            continue
        filtered_query.append((key, item))
    query = urlencode(sorted(filtered_query), doseq=True)
    path = parsed.path or "/"
    return urlunsplit((scheme, host_display, path, query, ""))


def _clean_domain(value: str) -> str:
    if not isinstance(value, str):
        raise InformationValidationError("allowed domain must be a string")
    candidate = value.strip().lower().rstrip(".")
    if "://" in candidate:
        parsed = urlsplit(candidate)
        candidate = parsed.hostname or ""
    candidate = candidate.lstrip(".")
    try:
        candidate = candidate.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise InformationValidationError("allowed domain is invalid") from exc
    if not candidate or len(candidate) > 253 or "/" in candidate or ":" in candidate:
        raise InformationValidationError("allowed domain is invalid")
    return candidate


def normalize_allowed_domains(
    values: Sequence[str] | None, *, base_url: str
) -> tuple[str, ...]:
    base_host = urlsplit(canonicalize_url(base_url)).hostname
    if base_host is None:
        raise InformationValidationError("base URL has no host")
    domains = {_clean_domain(base_host)}
    if values is not None:
        if isinstance(values, (str, bytes)):
            raise InformationValidationError("allowed_domains must be a sequence")
        domains.update(_clean_domain(value) for value in values)
    return tuple(sorted(domains))


def content_sha256(content: str) -> str:
    if not isinstance(content, str) or not content.strip():
        raise InformationValidationError("raw content cannot be empty")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _timestamp(value: str | datetime | None, field: str, *, optional: bool) -> str | None:
    if value is None:
        if optional:
            return None
        raise InformationValidationError(f"{field} is required")
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise InformationValidationError(f"{field} must be an ISO-8601 timestamp") from exc
    else:
        raise InformationValidationError(f"{field} must be a timestamp")
    if parsed.tzinfo is None:
        raise InformationValidationError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _clean_text(
    value: str | None,
    field: str,
    *,
    maximum: int,
    optional: bool = False,
) -> str | None:
    if value is None:
        if optional:
            return None
        raise InformationValidationError(f"{field} is required")
    if not isinstance(value, str):
        raise InformationValidationError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned and optional:
        return None
    if not cleaned or len(cleaned) > maximum:
        raise InformationValidationError(f"{field} length must be 1..{maximum}")
    return cleaned


def _positive_id(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InformationValidationError(f"{field} must be a positive integer")
    return value


def _enum(value: str, field: str, allowed: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise InformationValidationError(f"{field} is unsupported")
    return value


def _row_to_source(row: sqlite3.Row) -> InformationSource:
    try:
        decoded = json.loads(str(row["allowed_domains_json"]))
    except (TypeError, ValueError) as exc:
        raise InformationSchemaError("information source has invalid allowed_domains_json") from exc
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise InformationSchemaError("information source allowed domains must be a JSON array")
    domains = normalize_allowed_domains(decoded, base_url=str(row["base_url"]))
    return InformationSource(
        id=int(row["id"]),
        name=str(row["name"]),
        source_type=str(row["source_type"]),
        base_url=str(row["base_url"]),
        allowed_domains=domains,
        trust_level=int(row["trust_level"]),
        fetch_interval_minutes=int(row["fetch_interval_minutes"]),
        enabled=bool(row["enabled"]),
        last_checked_at=None if row["last_checked_at"] is None else str(row["last_checked_at"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_document(row: sqlite3.Row) -> SourceDocument:
    return SourceDocument(
        id=int(row["id"]),
        source_id=int(row["source_id"]),
        canonical_url=str(row["canonical_url"]),
        title=None if row["title"] is None else str(row["title"]),
        raw_content=str(row["raw_content"]),
        content_hash=str(row["content_hash"]),
        published_at=None if row["published_at"] is None else str(row["published_at"]),
        fetched_at=str(row["fetched_at"]),
        last_modified_at=None if row["last_modified_at"] is None else str(row["last_modified_at"]),
    )


def _row_to_update(row: sqlite3.Row) -> OfficialUpdate:
    return OfficialUpdate(
        id=int(row["id"]),
        document_id=int(row["document_id"]),
        source_id=int(row["source_id"]),
        source_name=str(row["source_name"]),
        canonical_url=str(row["canonical_url"]),
        published_at=None if row["published_at"] is None else str(row["published_at"]),
        category=str(row["category"]),
        title=str(row["title"]),
        summary=None if row["summary"] is None else str(row["summary"]),
        event_start_at=None if row["event_start_at"] is None else str(row["event_start_at"]),
        event_end_at=None if row["event_end_at"] is None else str(row["event_end_at"]),
        venue=None if row["venue"] is None else str(row["venue"]),
        status=str(row["status"]),
        verification_status=str(row["verification_status"]),
        confidence=float(row["confidence"]),
        replaces_update_id=None if row["replaces_update_id"] is None else int(row["replaces_update_id"]),
        reviewed_by=None if row["reviewed_by"] is None else str(row["reviewed_by"]),
        reviewed_at=None if row["reviewed_at"] is None else str(row["reviewed_at"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_run(row: sqlite3.Row) -> CollectorRun:
    keys = set(row.keys())
    return CollectorRun(
        id=int(row["id"]),
        source_id=int(row["source_id"]),
        started_at=str(row["started_at"]),
        finished_at=None if row["finished_at"] is None else str(row["finished_at"]),
        status=str(row["status"]),
        discovered_count=int(row["discovered_count"]),
        fetched_count=int(row["fetched_count"]),
        new_document_count=int(row["new_document_count"]),
        pending_update_count=int(row["pending_update_count"]),
        error_code=None if row["error_code"] is None else str(row["error_code"]),
        detail=None if row["detail"] is None else str(row["detail"]),
        source_name=(
            None
            if "source_name" not in keys or row["source_name"] is None
            else str(row["source_name"])
        ),
    )


def add_information_source(
    db_path: DatabasePath,
    *,
    name: str,
    source_type: str,
    base_url: str,
    allowed_domains: Sequence[str] | None = None,
    trust_level: int = 100,
    fetch_interval_minutes: int = 60,
    enabled: bool = True,
) -> InformationSource:
    clean_name = _clean_text(name, "name", maximum=200)
    clean_type = _enum(source_type, "source_type", SOURCE_TYPES)
    clean_url = canonicalize_url(base_url)
    domains = normalize_allowed_domains(allowed_domains, base_url=clean_url)
    if isinstance(trust_level, bool) or not isinstance(trust_level, int) or not 0 <= trust_level <= 100:
        raise InformationValidationError("trust_level must be an integer from 0 to 100")
    if (
        isinstance(fetch_interval_minutes, bool)
        or not isinstance(fetch_interval_minutes, int)
        or fetch_interval_minutes < 1
    ):
        raise InformationValidationError("fetch_interval_minutes must be positive")
    if not isinstance(enabled, bool):
        raise InformationValidationError("enabled must be boolean")
    now = _utc_now()
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            try:
                cursor = conn.execute(
                    "INSERT INTO information_sources "
                    "(name, source_type, base_url, allowed_domains_json, trust_level, "
                    "fetch_interval_minutes, enabled, last_checked_at, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                    (
                        clean_name,
                        clean_type,
                        clean_url,
                        json.dumps(domains, ensure_ascii=True, separators=(",", ":")),
                        trust_level,
                        fetch_interval_minutes,
                        int(enabled),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise InformationValidationError("source base URL already exists") from exc
            row = conn.execute(
                "SELECT * FROM information_sources WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:
                raise InformationStoreError("source disappeared during insert")
            result = _row_to_source(row)
        return result
    finally:
        conn.close()


def get_information_source(db_path: DatabasePath, source_id: int) -> InformationSource:
    clean_id = _positive_id(source_id, "source_id")
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM information_sources WHERE id = ?", (clean_id,)
        ).fetchone()
        if row is None:
            raise InformationNotFoundError("information source does not exist")
        return _row_to_source(row)
    finally:
        conn.close()


def list_information_sources(
    db_path: DatabasePath, *, enabled_only: bool = False
) -> list[InformationSource]:
    if not isinstance(enabled_only, bool):
        raise InformationValidationError("enabled_only must be boolean")
    conn = _connect(db_path)
    try:
        sql = "SELECT * FROM information_sources"
        params: tuple[object, ...] = ()
        if enabled_only:
            sql += " WHERE enabled = ?"
            params = (1,)
        rows = conn.execute(sql + " ORDER BY id", params).fetchall()
        return [_row_to_source(row) for row in rows]
    finally:
        conn.close()


def _validate_admin_audit_table(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "admin_audit_log"):
        return False
    rows = conn.execute("PRAGMA table_info(admin_audit_log)").fetchall()
    columns = {str(row[1]) for row in rows}
    required = {"actor", "action", "target_username", "detail", "created_at"}
    missing = required - columns
    if missing:
        raise InformationSchemaError(
            f"admin_audit_log missing columns: {', '.join(sorted(missing))}"
        )
    return True


def _optional_audit(
    conn: sqlite3.Connection, *, actor: str, action: str, detail: dict[str, object]
) -> None:
    clean_actor = _clean_text(actor, "actor", maximum=256)
    if not _validate_admin_audit_table(conn):
        return
    encoded = json.dumps(detail, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded) > 4_000:
        raise InformationValidationError("audit detail is too long")
    conn.execute(
        "INSERT INTO admin_audit_log "
        "(actor, action, target_username, detail, created_at) VALUES (?, ?, NULL, ?, ?)",
        (clean_actor, action, encoded, _utc_now()),
    )


def set_source_enabled(
    db_path: DatabasePath, source_id: int, enabled: bool, *, actor: str
) -> InformationSource:
    clean_id = _positive_id(source_id, "source_id")
    if not isinstance(enabled, bool):
        raise InformationValidationError("enabled must be boolean")
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            _validate_admin_audit_table(conn)
            cursor = conn.execute(
                "UPDATE information_sources SET enabled = ?, updated_at = ? WHERE id = ?",
                (int(enabled), _utc_now(), clean_id),
            )
            if cursor.rowcount != 1:
                raise InformationNotFoundError("information source does not exist")
            _optional_audit(
                conn,
                actor=actor,
                action="information.source_enabled" if enabled else "information.source_disabled",
                detail={"source_id": clean_id},
            )
            row = conn.execute(
                "SELECT * FROM information_sources WHERE id = ?", (clean_id,)
            ).fetchone()
            result = _row_to_source(row)
        return result
    finally:
        conn.close()


def mark_source_checked(
    db_path: DatabasePath, source_id: int, checked_at: str | datetime | None = None
) -> None:
    clean_id = _positive_id(source_id, "source_id")
    timestamp = _utc_now() if checked_at is None else _timestamp(checked_at, "checked_at", optional=False)
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            cursor = conn.execute(
                "UPDATE information_sources SET last_checked_at = ?, updated_at = ? WHERE id = ?",
                (timestamp, timestamp, clean_id),
            )
            if cursor.rowcount != 1:
                raise InformationNotFoundError("information source does not exist")
    finally:
        conn.close()


def find_source_document_by_hash(
    db_path: DatabasePath, source_id: int, content_hash: str
) -> SourceDocument | None:
    clean_source_id = _positive_id(source_id, "source_id")
    if not isinstance(content_hash, str) or _SHA256_PATTERN.fullmatch(content_hash) is None:
        raise InformationValidationError("content_hash must be a lowercase SHA-256 hex digest")
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM source_documents WHERE source_id = ? AND content_hash = ?",
            (clean_source_id, content_hash),
        ).fetchone()
        return None if row is None else _row_to_document(row)
    finally:
        conn.close()


def store_source_document(
    db_path: DatabasePath,
    *,
    source_id: int,
    canonical_url: str,
    raw_content: str,
    content_hash: str | None = None,
    title: str | None = None,
    published_at: str | datetime | None = None,
    fetched_at: str | datetime | None = None,
    last_modified_at: str | datetime | None = None,
) -> tuple[SourceDocument, bool]:
    clean_source_id = _positive_id(source_id, "source_id")
    clean_url = canonicalize_url(canonical_url)
    clean_content = _clean_text(raw_content, "raw_content", maximum=2_000_000)
    clean_hash = content_sha256(clean_content) if content_hash is None else content_hash
    if not isinstance(clean_hash, str) or _SHA256_PATTERN.fullmatch(clean_hash) is None:
        raise InformationValidationError("content_hash must be a lowercase SHA-256 hex digest")
    if clean_hash != content_sha256(clean_content):
        raise InformationValidationError("content_hash does not match raw_content")
    clean_title = _clean_text(title, "title", maximum=500, optional=True)
    clean_published = _timestamp(published_at, "published_at", optional=True)
    clean_fetched = _utc_now() if fetched_at is None else _timestamp(fetched_at, "fetched_at", optional=False)
    clean_modified = _timestamp(last_modified_at, "last_modified_at", optional=True)
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            if conn.execute(
                "SELECT 1 FROM information_sources WHERE id = ?", (clean_source_id,)
            ).fetchone() is None:
                raise InformationNotFoundError("information source does not exist")
            existing = conn.execute(
                "SELECT * FROM source_documents WHERE source_id = ? AND content_hash = ?",
                (clean_source_id, clean_hash),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    "UPDATE source_documents SET fetched_at = ?, "
                    "title = COALESCE(title, ?), published_at = COALESCE(published_at, ?), "
                    "last_modified_at = COALESCE(?, last_modified_at) WHERE id = ?",
                    (clean_fetched, clean_title, clean_published, clean_modified, int(existing["id"])),
                )
                row = conn.execute(
                    "SELECT * FROM source_documents WHERE id = ?", (int(existing["id"]),)
                ).fetchone()
                return _row_to_document(row), False
            cursor = conn.execute(
                "INSERT INTO source_documents "
                "(source_id, canonical_url, title, raw_content, content_hash, published_at, "
                "fetched_at, last_modified_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    clean_source_id,
                    clean_url,
                    clean_title,
                    clean_content,
                    clean_hash,
                    clean_published,
                    clean_fetched,
                    clean_modified,
                ),
            )
            row = conn.execute(
                "SELECT * FROM source_documents WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            return _row_to_document(row), True
    finally:
        conn.close()


def create_pending_update(
    db_path: DatabasePath,
    *,
    document_id: int,
    category: str,
    title: str,
    confidence: float,
    summary: str | None = None,
    event_start_at: str | datetime | None = None,
    event_end_at: str | datetime | None = None,
    venue: str | None = None,
    status: str = "scheduled",
    replaces_update_id: int | None = None,
) -> tuple[OfficialUpdate, bool]:
    clean_document_id = _positive_id(document_id, "document_id")
    clean_category = _enum(category, "category", UPDATE_CATEGORIES)
    clean_title = _clean_text(title, "title", maximum=500)
    clean_summary = _clean_text(summary, "summary", maximum=4_000, optional=True)
    clean_start = _timestamp(event_start_at, "event_start_at", optional=True)
    clean_end = _timestamp(event_end_at, "event_end_at", optional=True)
    if clean_start is not None and clean_end is not None and clean_end < clean_start:
        raise InformationValidationError("event_end_at cannot precede event_start_at")
    clean_venue = _clean_text(venue, "venue", maximum=500, optional=True)
    clean_status = _enum(status, "status", UPDATE_STATUSES)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise InformationValidationError("confidence must be numeric")
    clean_confidence = float(confidence)
    if not 0.0 <= clean_confidence <= 1.0:
        raise InformationValidationError("confidence must be between 0 and 1")
    clean_replaces = None if replaces_update_id is None else _positive_id(replaces_update_id, "replaces_update_id")
    now = _utc_now()
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            existing = conn.execute(
                _OFFICIAL_SELECT + "WHERE u.document_id = ?", (clean_document_id,)
            ).fetchone()
            if existing is not None:
                return _row_to_update(existing), False
            if conn.execute(
                "SELECT 1 FROM source_documents WHERE id = ?", (clean_document_id,)
            ).fetchone() is None:
                raise InformationNotFoundError("source document does not exist")
            if clean_replaces is not None and conn.execute(
                "SELECT 1 FROM official_updates WHERE id = ?", (clean_replaces,)
            ).fetchone() is None:
                raise InformationNotFoundError("replacement target does not exist")
            cursor = conn.execute(
                "INSERT INTO official_updates "
                "(document_id, category, title, summary, event_start_at, event_end_at, "
                "venue, status, verification_status, confidence, replaces_update_id, "
                "reviewed_by, reviewed_at, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL, NULL, ?, ?)",
                (
                    clean_document_id,
                    clean_category,
                    clean_title,
                    clean_summary,
                    clean_start,
                    clean_end,
                    clean_venue,
                    clean_status,
                    clean_confidence,
                    clean_replaces,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                _OFFICIAL_SELECT + "WHERE u.id = ?", (cursor.lastrowid,)
            ).fetchone()
            return _row_to_update(row), True
    finally:
        conn.close()


def get_official_update(db_path: DatabasePath, update_id: int) -> OfficialUpdate:
    clean_id = _positive_id(update_id, "update_id")
    conn = _connect(db_path)
    try:
        row = conn.execute(_OFFICIAL_SELECT + "WHERE u.id = ?", (clean_id,)).fetchone()
        if row is None:
            raise InformationNotFoundError("official update does not exist")
        return _row_to_update(row)
    finally:
        conn.close()


def list_official_updates(
    db_path: DatabasePath,
    *,
    verification_status: str | None = None,
    limit: int = 100,
) -> list[OfficialUpdate]:
    clean_status = (
        None
        if verification_status is None
        else _enum(
            verification_status,
            "verification_status",
            VERIFICATION_STATUSES,
        )
    )
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise InformationValidationError("limit must be an integer from 1 to 500")
    sql = _OFFICIAL_SELECT
    params: tuple[object, ...] = ()
    if clean_status is not None:
        sql += "WHERE u.verification_status = ? "
        params = (clean_status,)
    sql += "ORDER BY u.id DESC LIMIT ?"
    params += (limit,)
    conn = _connect(db_path)
    try:
        return [_row_to_update(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def find_latest_update_by_canonical_url(
    db_path: DatabasePath, canonical_url: str
) -> OfficialUpdate | None:
    clean_url = canonicalize_url(canonical_url)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            _OFFICIAL_SELECT + "WHERE d.canonical_url = ? ORDER BY u.id DESC LIMIT 1",
            (clean_url,),
        ).fetchone()
        return None if row is None else _row_to_update(row)
    finally:
        conn.close()


def review_official_update(
    db_path: DatabasePath,
    update_id: int,
    decision: str,
    *,
    actor: str,
    reason: str | None = None,
) -> OfficialUpdate:
    clean_id = _positive_id(update_id, "update_id")
    clean_decision = _enum(decision, "decision", ("approved", "rejected"))
    clean_actor = _clean_text(actor, "actor", maximum=256)
    clean_reason = _clean_text(reason, "reason", maximum=1_000, optional=True)
    now = _utc_now()
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            _validate_admin_audit_table(conn)
            row = conn.execute(
                "SELECT verification_status FROM official_updates WHERE id = ?", (clean_id,)
            ).fetchone()
            if row is None:
                raise InformationNotFoundError("official update does not exist")
            if str(row["verification_status"]) != "pending":
                raise InformationValidationError("only pending updates can be reviewed")
            cursor = conn.execute(
                "UPDATE official_updates SET verification_status = ?, reviewed_by = ?, "
                "reviewed_at = ?, updated_at = ? WHERE id = ? AND verification_status = 'pending'",
                (clean_decision, clean_actor, now, now, clean_id),
            )
            if cursor.rowcount != 1:
                raise InformationStoreError("review update lost a concurrent race")
            detail: dict[str, object] = {"update_id": clean_id}
            if clean_reason is not None:
                detail["reason"] = clean_reason
            _optional_audit(
                conn,
                actor=clean_actor,
                action=f"information.update_{clean_decision}",
                detail=detail,
            )
            result_row = conn.execute(
                _OFFICIAL_SELECT + "WHERE u.id = ?", (clean_id,)
            ).fetchone()
            result = _row_to_update(result_row)
        return result
    finally:
        conn.close()


def revoke_official_update(
    db_path: DatabasePath,
    update_id: int,
    *,
    actor: str,
    reason: str,
) -> OfficialUpdate:
    """Withdraw an approved update so it can no longer ground chat replies.

    Revocation is intentionally separate from the initial review operation:
    only approved records may be revoked, and a reason is mandatory so the
    administrative audit entry remains useful.
    """

    clean_id = _positive_id(update_id, "update_id")
    clean_actor = _clean_text(actor, "actor", maximum=256)
    clean_reason = _clean_text(reason, "reason", maximum=1_000)
    now = _utc_now()
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            _validate_admin_audit_table(conn)
            row = conn.execute(
                "SELECT verification_status FROM official_updates WHERE id = ?", (clean_id,)
            ).fetchone()
            if row is None:
                raise InformationNotFoundError("official update does not exist")
            if str(row["verification_status"]) != "approved":
                raise InformationValidationError("only approved updates can be revoked")
            cursor = conn.execute(
                "UPDATE official_updates SET verification_status = 'rejected', "
                "reviewed_by = ?, reviewed_at = ?, updated_at = ? "
                "WHERE id = ? AND verification_status = 'approved'",
                (clean_actor, now, now, clean_id),
            )
            if cursor.rowcount != 1:
                raise InformationStoreError("revoke update lost a concurrent race")
            _optional_audit(
                conn,
                actor=clean_actor,
                action="information.update_revoked",
                detail={"update_id": clean_id, "reason": clean_reason},
            )
            result_row = conn.execute(
                _OFFICIAL_SELECT + "WHERE u.id = ?", (clean_id,)
            ).fetchone()
            result = _row_to_update(result_row)
        return result
    finally:
        conn.close()


def query_recent_approved_updates(
    db_path: DatabasePath,
    *,
    now: str | datetime | None = None,
    days_back: int = 7,
    days_ahead: int = 90,
    limit: int = 50,
    categories: Sequence[str] | None = None,
) -> list[OfficialUpdate]:
    if isinstance(days_back, bool) or not isinstance(days_back, int) or not 0 <= days_back <= 3650:
        raise InformationValidationError("days_back must be an integer from 0 to 3650")
    if isinstance(days_ahead, bool) or not isinstance(days_ahead, int) or not 0 <= days_ahead <= 3650:
        raise InformationValidationError("days_ahead must be an integer from 0 to 3650")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise InformationValidationError("limit must be an integer from 1 to 500")
    center_text = _utc_now() if now is None else _timestamp(now, "now", optional=False)
    center = datetime.fromisoformat(center_text.replace("Z", "+00:00"))
    lower = _timestamp(center - timedelta(days=days_back), "lower", optional=False)
    upper = _timestamp(center + timedelta(days=days_ahead), "upper", optional=False)
    clean_categories: tuple[str, ...] = ()
    if categories is not None:
        if isinstance(categories, (str, bytes)):
            raise InformationValidationError("categories must be a sequence")
        clean_categories = tuple(_enum(item, "category", UPDATE_CATEGORIES) for item in categories)
    sql = (
        "WITH RECURSIVE replacement_ancestors(ancestor_id) AS ("
        "SELECT replaces_update_id FROM official_updates "
        "WHERE verification_status = 'approved' AND replaces_update_id IS NOT NULL "
        "UNION "
        "SELECT parent.replaces_update_id FROM official_updates AS parent "
        "JOIN replacement_ancestors AS chain ON parent.id = chain.ancestor_id "
        "WHERE parent.replaces_update_id IS NOT NULL"
        ") "
        + _OFFICIAL_SELECT
        + "WHERE u.verification_status = 'approved' "
        "AND u.id NOT IN (SELECT ancestor_id FROM replacement_ancestors) "
        "AND COALESCE(u.event_start_at, d.published_at, u.created_at) BETWEEN ? AND ? "
    )
    params: list[object] = [lower, upper]
    if clean_categories:
        sql += "AND u.category IN (" + ",".join("?" for _ in clean_categories) + ") "
        params.extend(clean_categories)
    sql += (
        "ORDER BY COALESCE(u.event_start_at, d.published_at, u.created_at) ASC, u.id ASC "
        "LIMIT ?"
    )
    params.append(limit)
    conn = _connect(db_path)
    try:
        return [_row_to_update(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def start_collector_run(
    db_path: DatabasePath,
    source_id: int,
    *,
    started_at: str | datetime | None = None,
) -> CollectorRun:
    clean_source_id = _positive_id(source_id, "source_id")
    clean_started = _utc_now() if started_at is None else _timestamp(started_at, "started_at", optional=False)
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            if conn.execute(
                "SELECT 1 FROM information_sources WHERE id = ?", (clean_source_id,)
            ).fetchone() is None:
                raise InformationNotFoundError("information source does not exist")
            cursor = conn.execute(
                "INSERT INTO collector_runs "
                "(source_id, started_at, finished_at, status, discovered_count, fetched_count, "
                "new_document_count, pending_update_count, error_code, detail) "
                "VALUES (?, ?, NULL, 'running', 0, 0, 0, 0, NULL, NULL)",
                (clean_source_id, clean_started),
            )
            row = conn.execute(
                "SELECT * FROM collector_runs WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            return _row_to_run(row)
    finally:
        conn.close()


def finish_collector_run(
    db_path: DatabasePath,
    run_id: int,
    *,
    status: str,
    discovered_count: int,
    fetched_count: int,
    new_document_count: int,
    pending_update_count: int,
    error_code: str | None = None,
    detail: str | None = None,
    finished_at: str | datetime | None = None,
) -> CollectorRun:
    clean_id = _positive_id(run_id, "run_id")
    clean_status = _enum(status, "status", ("succeeded", "partial", "failed"))
    counts = []
    for field, value in (
        ("discovered_count", discovered_count),
        ("fetched_count", fetched_count),
        ("new_document_count", new_document_count),
        ("pending_update_count", pending_update_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InformationValidationError(f"{field} must be a non-negative integer")
        counts.append(value)
    clean_error = _clean_text(error_code, "error_code", maximum=120, optional=True)
    clean_detail = _clean_text(detail, "detail", maximum=4_000, optional=True)
    clean_finished = _utc_now() if finished_at is None else _timestamp(finished_at, "finished_at", optional=False)
    conn = _connect(db_path)
    try:
        with _immediate_transaction(conn):
            row = conn.execute(
                "SELECT status FROM collector_runs WHERE id = ?", (clean_id,)
            ).fetchone()
            if row is None:
                raise InformationNotFoundError("collector run does not exist")
            if str(row["status"]) != "running":
                raise InformationValidationError("collector run is already finished")
            conn.execute(
                "UPDATE collector_runs SET finished_at = ?, status = ?, discovered_count = ?, "
                "fetched_count = ?, new_document_count = ?, pending_update_count = ?, "
                "error_code = ?, detail = ? WHERE id = ?",
                (
                    clean_finished,
                    clean_status,
                    *counts,
                    clean_error,
                    clean_detail,
                    clean_id,
                ),
            )
            result = conn.execute(
                "SELECT * FROM collector_runs WHERE id = ?", (clean_id,)
            ).fetchone()
            return _row_to_run(result)
    finally:
        conn.close()


def list_collector_runs(
    db_path: DatabasePath,
    *,
    source_id: int | None = None,
    limit: int = 100,
) -> list[CollectorRun]:
    clean_source_id = (
        None if source_id is None else _positive_id(source_id, "source_id")
    )
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise InformationValidationError("limit must be an integer from 1 to 500")
    sql = (
        "SELECT r.*, s.name AS source_name FROM collector_runs AS r "
        "JOIN information_sources AS s ON s.id = r.source_id "
    )
    params: tuple[object, ...] = ()
    if clean_source_id is not None:
        sql += "WHERE r.source_id = ? "
        params = (clean_source_id,)
    sql += "ORDER BY r.id DESC LIMIT ?"
    params += (limit,)
    conn = _connect(db_path)
    try:
        return [_row_to_run(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
