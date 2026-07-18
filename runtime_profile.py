"""Explicit, user-scoped runtime context preferences.

This module stores only values a signed-in user or their browser explicitly
provided. It never derives a location from an IP address or silently upgrades
coarse coordinates into a precise location history.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypeAlias
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DatabasePath: TypeAlias = str | Path

BUSY_TIMEOUT_MS = 5_000
MAX_CITY_LENGTH = 120
MAX_LOCALE_LENGTH = 35
MAX_TIMEZONE_LENGTH = 64
COARSE_COORDINATE_DECIMALS = 2
MAX_TEMPORARY_CITY_LIFETIME = timedelta(days=90)
MAX_COORDINATE_LIFETIME = timedelta(days=7)

_TABLE_NAME = "user_runtime_profiles"
_SELECT_COLUMNS = (
    "username, browser_timezone, browser_locale, home_city, temporary_city, "
    "temporary_city_expires_at, coarse_latitude, coarse_longitude, "
    "coarse_coordinates_expires_at, created_at, updated_at"
)
_LOCALE_PATTERN = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
    username TEXT PRIMARY KEY,
    browser_timezone TEXT,
    browser_locale TEXT,
    home_city TEXT,
    temporary_city TEXT,
    temporary_city_expires_at TEXT,
    coarse_latitude REAL,
    coarse_longitude REAL,
    coarse_coordinates_expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE,
    CHECK (browser_timezone IS NULL OR length(browser_timezone) BETWEEN 1 AND {MAX_TIMEZONE_LENGTH}),
    CHECK (browser_locale IS NULL OR length(browser_locale) BETWEEN 1 AND {MAX_LOCALE_LENGTH}),
    CHECK (home_city IS NULL OR length(home_city) BETWEEN 1 AND {MAX_CITY_LENGTH}),
    CHECK (
        (temporary_city IS NULL AND temporary_city_expires_at IS NULL)
        OR
        (temporary_city IS NOT NULL AND temporary_city_expires_at IS NOT NULL
         AND length(temporary_city) BETWEEN 1 AND {MAX_CITY_LENGTH})
    ),
    CHECK (
        (coarse_latitude IS NULL AND coarse_longitude IS NULL
         AND coarse_coordinates_expires_at IS NULL)
        OR
        (coarse_latitude BETWEEN -90.0 AND 90.0
         AND coarse_longitude BETWEEN -180.0 AND 180.0
         AND coarse_coordinates_expires_at IS NOT NULL)
    )
)
"""

_EXPECTED_COLUMNS = (
    ("username", "TEXT", 0, None, 1),
    ("browser_timezone", "TEXT", 0, None, 0),
    ("browser_locale", "TEXT", 0, None, 0),
    ("home_city", "TEXT", 0, None, 0),
    ("temporary_city", "TEXT", 0, None, 0),
    ("temporary_city_expires_at", "TEXT", 0, None, 0),
    ("coarse_latitude", "REAL", 0, None, 0),
    ("coarse_longitude", "REAL", 0, None, 0),
    ("coarse_coordinates_expires_at", "TEXT", 0, None, 0),
    ("created_at", "TEXT", 1, None, 0),
    ("updated_at", "TEXT", 1, None, 0),
)

_REQUIRED_TABLE_SQL_FRAGMENTS = (
    "usernametextprimarykey",
    "foreignkey(username)referencesusers(username)ondeletecascade",
    "temporary_cityisnullandtemporary_city_expires_atisnull",
    "temporary_cityisnotnullandtemporary_city_expires_atisnotnull",
    "coarse_latitudeisnullandcoarse_longitudeisnullandcoarse_coordinates_expires_atisnull",
    "coarse_latitudebetween-90.0and90.0",
    "coarse_longitudebetween-180.0and180.0",
)


class RuntimeProfileError(RuntimeError):
    """Base exception for runtime-profile operations."""


class RuntimeProfileSchemaError(RuntimeProfileError):
    """Raised when an existing runtime-profile schema is incompatible."""


class RuntimeProfileDataError(RuntimeProfileError):
    """Raised when persisted profile data cannot be interpreted safely."""


class RuntimeProfileValidationError(ValueError):
    """Raised when a profile mutation violates the public contract."""


@dataclass(frozen=True)
class RuntimeLocation:
    """One explicitly supplied location suitable for a weather lookup."""

    kind: str
    city: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    expires_at: datetime | None = None

    def to_dict(self, *, include_coordinates: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {"kind": self.kind}
        if self.city is not None:
            payload["city"] = self.city
        if include_coordinates and self.latitude is not None and self.longitude is not None:
            payload["latitude"] = self.latitude
            payload["longitude"] = self.longitude
        if self.expires_at is not None:
            payload["expires_at"] = _format_utc(self.expires_at)
        return payload


@dataclass(frozen=True)
class RuntimeProfile:
    username: str
    browser_timezone: str | None
    browser_locale: str | None
    home_city: str | None
    temporary_city: str | None
    temporary_city_expires_at: datetime | None
    coarse_latitude: float | None
    coarse_longitude: float | None
    coarse_coordinates_expires_at: datetime | None
    created_at: datetime
    updated_at: datetime

    def effective_location(self) -> RuntimeLocation | None:
        """Return only an explicit active location, never an inferred one."""

        if self.temporary_city is not None:
            return RuntimeLocation(
                kind="temporary_city",
                city=self.temporary_city,
                expires_at=self.temporary_city_expires_at,
            )
        if self.coarse_latitude is not None and self.coarse_longitude is not None:
            return RuntimeLocation(
                kind="authorized_coarse_coordinates",
                latitude=self.coarse_latitude,
                longitude=self.coarse_longitude,
                expires_at=self.coarse_coordinates_expires_at,
            )
        if self.home_city is not None:
            return RuntimeLocation(kind="home_city", city=self.home_city)
        return None


def _compact_schema_sql(value: str) -> str:
    compact = re.sub(r"\s+", "", value.lower())
    return compact.translate(str.maketrans("", "", '"`[]'))


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


def _validate_table_schema(conn: sqlite3.Connection) -> None:
    rows = conn.execute(f"PRAGMA table_info({_TABLE_NAME})").fetchall()
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), row[4], int(row[5]))
        for row in rows
    )
    if actual_columns != _EXPECTED_COLUMNS:
        raise RuntimeProfileSchemaError(
            f"Incompatible {_TABLE_NAME} columns: expected {_EXPECTED_COLUMNS!r}, "
            f"got {actual_columns!r}"
        )

    schema_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (_TABLE_NAME,),
    ).fetchone()
    if schema_row is None or not schema_row[0]:
        raise RuntimeProfileSchemaError(f"Missing CREATE TABLE SQL for {_TABLE_NAME}")
    compact_sql = _compact_schema_sql(str(schema_row[0]))
    missing_fragments = [
        fragment for fragment in _REQUIRED_TABLE_SQL_FRAGMENTS if fragment not in compact_sql
    ]
    if missing_fragments:
        raise RuntimeProfileSchemaError(
            f"Incompatible {_TABLE_NAME} constraints: missing {missing_fragments!r}"
        )

    foreign_keys = conn.execute(f"PRAGMA foreign_key_list({_TABLE_NAME})").fetchall()
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
    expected_foreign_key = (
        "users",
        "username",
        "username",
        "NO ACTION",
        "CASCADE",
        "NONE",
    )
    if actual_foreign_keys != {expected_foreign_key}:
        raise RuntimeProfileSchemaError(
            f"Incompatible {_TABLE_NAME} foreign keys: {actual_foreign_keys!r}"
        )


def init_runtime_profile_schema(conn: sqlite3.Connection) -> None:
    """Create and validate the additive schema without committing caller work."""

    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be a sqlite3.Connection")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
        raise RuntimeProfileSchemaError(
            "SQLite foreign_keys could not be enabled; initialize outside an active transaction"
        )
    if not _table_exists(conn, "users"):
        raise RuntimeProfileSchemaError("users table must exist before runtime profiles")
    conn.execute(_CREATE_TABLE_SQL)
    _validate_table_schema(conn)


def _connect(db_path: DatabasePath) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_MS / 1_000)
    try:
        init_runtime_profile_schema(conn)
        conn.commit()
    except Exception:
        conn.close()
        raise
    return conn


def _clean_username(username: str) -> str:
    if not isinstance(username, str):
        raise RuntimeProfileValidationError("username must be a string")
    cleaned = username.strip()
    if not cleaned:
        raise RuntimeProfileValidationError("username cannot be empty")
    return cleaned


def _clean_optional_city(city: str | None, field_name: str) -> str | None:
    if city is None:
        return None
    if not isinstance(city, str):
        raise RuntimeProfileValidationError(f"{field_name} must be a string or None")
    cleaned = city.strip()
    if not 1 <= len(cleaned) <= MAX_CITY_LENGTH:
        raise RuntimeProfileValidationError(
            f"{field_name} length must be 1..{MAX_CITY_LENGTH}"
        )
    if _CONTROL_CHARACTER_PATTERN.search(cleaned):
        raise RuntimeProfileValidationError(f"{field_name} cannot contain control characters")
    return cleaned


def _clean_timezone_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeProfileValidationError("browser_timezone must be a string or None")
    cleaned = value.strip()
    if not 1 <= len(cleaned) <= MAX_TIMEZONE_LENGTH:
        raise RuntimeProfileValidationError(
            f"browser_timezone length must be 1..{MAX_TIMEZONE_LENGTH}"
        )
    if _CONTROL_CHARACTER_PATTERN.search(cleaned):
        raise RuntimeProfileValidationError(
            "browser_timezone cannot contain control characters"
        )
    try:
        ZoneInfo(cleaned)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise RuntimeProfileValidationError("browser_timezone is not a valid IANA timezone") from exc
    return cleaned


def _clean_locale(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeProfileValidationError("browser_locale must be a string or None")
    cleaned = value.strip().replace("_", "-")
    if not 1 <= len(cleaned) <= MAX_LOCALE_LENGTH or not _LOCALE_PATTERN.fullmatch(cleaned):
        raise RuntimeProfileValidationError("browser_locale must be a valid language tag")
    return cleaned


def _aware_utc(
    value: datetime,
    field_name: str,
    *,
    now: datetime,
    max_lifetime: timedelta,
) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RuntimeProfileValidationError(f"{field_name} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized <= now:
        raise RuntimeProfileValidationError(f"{field_name} must be in the future")
    if normalized - now > max_lifetime:
        raise RuntimeProfileValidationError(
            f"{field_name} is too far in the future"
        )
    return normalized


def _normalize_now(value: datetime | None = None) -> datetime:
    current = value if value is not None else datetime.now(timezone.utc)
    if not isinstance(current, datetime) or current.tzinfo is None:
        raise RuntimeProfileValidationError("now must be a timezone-aware datetime")
    return current.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RuntimeProfileDataError(f"{field_name} is not a valid timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeProfileDataError(f"{field_name} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise RuntimeProfileDataError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _clean_coordinate(value: float, field_name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeProfileValidationError(f"{field_name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise RuntimeProfileValidationError(
            f"{field_name} must be between {minimum:g} and {maximum:g}"
        )
    return round(numeric, COARSE_COORDINATE_DECIMALS)


def _run_upsert(
    db_path: DatabasePath,
    username: str,
    insert_columns: tuple[str, ...],
    values: tuple[object, ...],
    update_assignments: tuple[str, ...],
    *,
    now: datetime,
) -> RuntimeProfile:
    cleaned_username = _clean_username(username)
    timestamp = _format_utc(now)
    columns_sql = ", ".join(("username", *insert_columns, "created_at", "updated_at"))
    placeholders = ", ".join("?" for _ in range(len(insert_columns) + 3))
    updates_sql = ", ".join((*update_assignments, "updated_at = excluded.updated_at"))
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            f"INSERT INTO {_TABLE_NAME} ({columns_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT(username) DO UPDATE SET {updates_sql}",
            (cleaned_username, *values, timestamp, timestamp),
        )
        row = conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM {_TABLE_NAME} WHERE username = ?",
            (cleaned_username,),
        ).fetchone()
        if row is None:
            raise RuntimeError("runtime profile disappeared during upsert")
        conn.commit()
        return _row_to_profile(row)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        if "FOREIGN KEY constraint failed" in str(exc):
            raise RuntimeProfileValidationError("username does not exist") from exc
        raise RuntimeProfileValidationError("runtime profile violates storage constraints") from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _row_to_profile(row: tuple[object, ...]) -> RuntimeProfile:
    temporary_expires = (
        None if row[5] is None else _parse_utc(row[5], "temporary_city_expires_at")
    )
    coordinate_expires = (
        None
        if row[8] is None
        else _parse_utc(row[8], "coarse_coordinates_expires_at")
    )
    return RuntimeProfile(
        username=str(row[0]),
        browser_timezone=None if row[1] is None else str(row[1]),
        browser_locale=None if row[2] is None else str(row[2]),
        home_city=None if row[3] is None else str(row[3]),
        temporary_city=None if row[4] is None else str(row[4]),
        temporary_city_expires_at=temporary_expires,
        coarse_latitude=None if row[6] is None else float(row[6]),
        coarse_longitude=None if row[7] is None else float(row[7]),
        coarse_coordinates_expires_at=coordinate_expires,
        created_at=_parse_utc(row[9], "created_at"),
        updated_at=_parse_utc(row[10], "updated_at"),
    )


def set_browser_context(
    db_path: DatabasePath,
    username: str,
    browser_timezone: str | None,
    browser_locale: str | None,
    *,
    now: datetime | None = None,
) -> RuntimeProfile:
    """Store browser-reported timezone and locale after strict validation."""

    current = _normalize_now(now)
    clean_timezone = _clean_timezone_name(browser_timezone)
    clean_locale = _clean_locale(browser_locale)
    return _run_upsert(
        db_path,
        username,
        ("browser_timezone", "browser_locale"),
        (clean_timezone, clean_locale),
        (
            "browser_timezone = excluded.browser_timezone",
            "browser_locale = excluded.browser_locale",
        ),
        now=current,
    )


def set_home_city(
    db_path: DatabasePath,
    username: str,
    city: str,
    *,
    now: datetime | None = None,
) -> RuntimeProfile:
    """Store a manually selected, long-lived city."""

    current = _normalize_now(now)
    clean_city = _clean_optional_city(city, "home_city")
    if clean_city is None:
        raise RuntimeProfileValidationError("home_city must be a string")
    return _run_upsert(
        db_path,
        username,
        ("home_city",),
        (clean_city,),
        ("home_city = excluded.home_city",),
        now=current,
    )


def set_temporary_city(
    db_path: DatabasePath,
    username: str,
    city: str,
    expires_at: datetime,
    *,
    now: datetime | None = None,
) -> RuntimeProfile:
    """Store a manually selected current city with a bounded lifetime."""

    current = _normalize_now(now)
    clean_city = _clean_optional_city(city, "temporary_city")
    if clean_city is None:
        raise RuntimeProfileValidationError("temporary_city must be a string")
    clean_expiry = _aware_utc(
        expires_at,
        "temporary_city_expires_at",
        now=current,
        max_lifetime=MAX_TEMPORARY_CITY_LIFETIME,
    )
    return _run_upsert(
        db_path,
        username,
        (
            "temporary_city",
            "temporary_city_expires_at",
            "coarse_latitude",
            "coarse_longitude",
            "coarse_coordinates_expires_at",
        ),
        (clean_city, _format_utc(clean_expiry), None, None, None),
        (
            "temporary_city = excluded.temporary_city",
            "temporary_city_expires_at = excluded.temporary_city_expires_at",
            "coarse_latitude = NULL",
            "coarse_longitude = NULL",
            "coarse_coordinates_expires_at = NULL",
        ),
        now=current,
    )


def set_authorized_coordinates(
    db_path: DatabasePath,
    username: str,
    latitude: float,
    longitude: float,
    expires_at: datetime,
    *,
    now: datetime | None = None,
) -> RuntimeProfile:
    """Store user-authorized coordinates rounded to coarse city-level precision."""

    current = _normalize_now(now)
    clean_latitude = _clean_coordinate(latitude, "latitude", -90.0, 90.0)
    clean_longitude = _clean_coordinate(longitude, "longitude", -180.0, 180.0)
    clean_expiry = _aware_utc(
        expires_at,
        "coarse_coordinates_expires_at",
        now=current,
        max_lifetime=MAX_COORDINATE_LIFETIME,
    )
    return _run_upsert(
        db_path,
        username,
        (
            "temporary_city",
            "temporary_city_expires_at",
            "coarse_latitude",
            "coarse_longitude",
            "coarse_coordinates_expires_at",
        ),
        (None, None, clean_latitude, clean_longitude, _format_utc(clean_expiry)),
        (
            "temporary_city = NULL",
            "temporary_city_expires_at = NULL",
            "coarse_latitude = excluded.coarse_latitude",
            "coarse_longitude = excluded.coarse_longitude",
            "coarse_coordinates_expires_at = excluded.coarse_coordinates_expires_at",
        ),
        now=current,
    )


def _clear_fields(
    db_path: DatabasePath,
    username: str,
    assignments: str,
    *,
    now: datetime | None = None,
) -> bool:
    cleaned_username = _clean_username(username)
    current = _normalize_now(now)
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            f"UPDATE {_TABLE_NAME} SET {assignments}, updated_at = ? WHERE username = ?",
            (_format_utc(current), cleaned_username),
        )
        conn.commit()
        return cursor.rowcount == 1
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def clear_home_city(
    db_path: DatabasePath,
    username: str,
    *,
    now: datetime | None = None,
) -> bool:
    return _clear_fields(db_path, username, "home_city = NULL", now=now)


def clear_temporary_city(
    db_path: DatabasePath,
    username: str,
    *,
    now: datetime | None = None,
) -> bool:
    return _clear_fields(
        db_path,
        username,
        "temporary_city = NULL, temporary_city_expires_at = NULL",
        now=now,
    )


def clear_authorized_coordinates(
    db_path: DatabasePath,
    username: str,
    *,
    now: datetime | None = None,
) -> bool:
    return _clear_fields(
        db_path,
        username,
        (
            "coarse_latitude = NULL, coarse_longitude = NULL, "
            "coarse_coordinates_expires_at = NULL"
        ),
        now=now,
    )


def clear_runtime_profile(db_path: DatabasePath, username: str) -> bool:
    """Hard-delete every stored runtime preference for one user."""

    cleaned_username = _clean_username(username)
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            f"DELETE FROM {_TABLE_NAME} WHERE username = ?",
            (cleaned_username,),
        )
        conn.commit()
        return cursor.rowcount == 1
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_runtime_profile(
    db_path: DatabasePath,
    username: str,
    *,
    now: datetime | None = None,
) -> RuntimeProfile | None:
    """Return one profile after atomically removing any expired locations."""

    cleaned_username = _clean_username(username)
    current = _normalize_now(now)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM {_TABLE_NAME} WHERE username = ?",
            (cleaned_username,),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        profile = _row_to_profile(row)
        clear_temporary = (
            profile.temporary_city_expires_at is not None
            and profile.temporary_city_expires_at <= current
        )
        clear_coordinates = (
            profile.coarse_coordinates_expires_at is not None
            and profile.coarse_coordinates_expires_at <= current
        )
        if clear_temporary or clear_coordinates:
            assignments: list[str] = []
            if clear_temporary:
                assignments.extend(
                    ["temporary_city = NULL", "temporary_city_expires_at = NULL"]
                )
            if clear_coordinates:
                assignments.extend(
                    [
                        "coarse_latitude = NULL",
                        "coarse_longitude = NULL",
                        "coarse_coordinates_expires_at = NULL",
                    ]
                )
            assignments.append("updated_at = ?")
            conn.execute(
                f"UPDATE {_TABLE_NAME} SET {', '.join(assignments)} WHERE username = ?",
                (_format_utc(current), cleaned_username),
            )
            row = conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM {_TABLE_NAME} WHERE username = ?",
                (cleaned_username,),
            ).fetchone()
            if row is None:
                raise RuntimeError("runtime profile disappeared during expiry cleanup")
            profile = _row_to_profile(row)
        conn.commit()
        return profile
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
