"""Small, UI-independent helpers for account authentication and revocation."""

from __future__ import annotations

import sqlite3
import re
from pathlib import Path
from typing import Protocol, TypeAlias

from argon2.exceptions import InvalidHashError, VerificationError


DatabasePath: TypeAlias = str | Path


class AccountAuthSchemaError(RuntimeError):
    """Raised when the users table cannot safely support session revocation."""


class PasswordVerifier(Protocol):
    def verify(self, hash: str | bytes, password: str | bytes) -> bool: ...


def _users_column_rows(conn: sqlite3.Connection) -> dict[str, tuple[object, ...]]:
    table = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'"
    ).fetchone()
    if table is None:
        raise AccountAuthSchemaError("Required users table is missing")
    rows = {
        str(row[1]): row for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    missing = {"username", "password_hash"} - set(rows)
    if missing:
        raise AccountAuthSchemaError(
            f"users is missing required columns: {', '.join(sorted(missing))}"
        )
    if int(rows["username"][5]) != 1:
        raise AccountAuthSchemaError("users.username must be the primary key")
    if int(rows["password_hash"][3]) != 1:
        raise AccountAuthSchemaError("users.password_hash must be NOT NULL")
    return rows


def _validate_auth_version_schema(conn: sqlite3.Connection) -> None:
    rows = _users_column_rows(conn)
    row = rows.get("auth_version")
    if row is None:
        raise AccountAuthSchemaError("users.auth_version is missing")
    default = "" if row[4] is None else str(row[4]).strip("()'\" ")
    if (
        str(row[2]).upper() != "INTEGER"
        or int(row[3]) != 1
        or default != "1"
        or int(row[5]) != 0
    ):
        raise AccountAuthSchemaError("users.auth_version has an incompatible definition")

    create_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'"
    ).fetchone()
    compact_sql = re.sub(r"[\s\"`\[\]]+", "", str(create_row[0]).lower())
    if "check(auth_version>=1)" not in compact_sql:
        raise AccountAuthSchemaError("users.auth_version must enforce values >= 1")
    invalid = conn.execute(
        "SELECT 1 FROM users WHERE auth_version IS NULL "
        "OR typeof(auth_version) <> 'integer' OR auth_version < 1 LIMIT 1"
    ).fetchone()
    if invalid is not None:
        raise AccountAuthSchemaError("users contains invalid auth_version values")


def init_account_auth_schema(conn: sqlite3.Connection) -> None:
    """Atomically add and validate ``users.auth_version``.

    The caller owns the connection and transaction. When this helper starts a
    transaction it deliberately leaves it open so later startup migrations can
    commit or roll back as one unit.
    """

    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be a sqlite3.Connection")

    started_transaction = not conn.in_transaction
    savepoint = "account_auth_schema_migration"
    savepoint_open = False
    try:
        if started_transaction:
            conn.execute("BEGIN IMMEDIATE")
        conn.execute(f"SAVEPOINT {savepoint}")
        savepoint_open = True

        rows = _users_column_rows(conn)
        if "auth_version" not in rows:
            conn.execute(
                "ALTER TABLE users ADD COLUMN auth_version INTEGER NOT NULL "
                "DEFAULT 1 CHECK (auth_version >= 1)"
            )
        _validate_auth_version_schema(conn)
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


def verify_account(
    db_path: DatabasePath,
    username: str,
    password: str,
    password_verifier: PasswordVerifier,
) -> int | None:
    """Return the account's session version after a valid password check."""

    if not isinstance(username, str) or not username:
        return None
    if not isinstance(password, str) or not password:
        return None

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT password_hash, auth_version FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None

    try:
        password_matches = password_verifier.verify(str(row[0]), password)
        auth_version = int(row[1])
    except (InvalidHashError, VerificationError, TypeError, ValueError):
        return None
    return auth_version if password_matches and auth_version >= 1 else None


def get_account_auth_version(
    db_path: DatabasePath, username: str
) -> int | None:
    """Return the current revocation version, or None for a missing account."""

    if not isinstance(username, str) or not username:
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT auth_version FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        auth_version = int(row[0])
    except (TypeError, ValueError):
        return None
    return auth_version if auth_version >= 1 else None
