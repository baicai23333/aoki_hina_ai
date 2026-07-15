import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from argon2 import PasswordHasher

from account_auth import (
    AccountAuthSchemaError,
    get_account_auth_version,
    init_account_auth_schema,
    verify_account,
)


ROOT = Path(__file__).resolve().parents[1]


class AccountAuthTests(unittest.TestCase):
    def setUp(self):
        self.db_path = ROOT / f".test_account_auth_{uuid4().hex}.db"
        self.extra_paths = []
        self.hasher = PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "CREATE TABLE users ("
                "username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, "
                "auth_version INTEGER NOT NULL DEFAULT 1)"
            )
            conn.executemany(
                "INSERT INTO users (username, password_hash, auth_version) "
                "VALUES (?, ?, ?)",
                (
                    ("alice", self.hasher.hash("correct-password"), 3),
                    ("broken", "not-an-argon2-hash", 1),
                ),
            )
            conn.commit()

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)
        for path in self.extra_paths:
            path.unlink(missing_ok=True)

    def test_valid_password_returns_current_session_version(self):
        self.assertEqual(
            verify_account(
                self.db_path, "alice", "correct-password", self.hasher
            ),
            3,
        )
        self.assertEqual(get_account_auth_version(self.db_path, "alice"), 3)

    def test_wrong_password_missing_user_and_broken_hash_fail_closed(self):
        self.assertIsNone(
            verify_account(self.db_path, "alice", "wrong", self.hasher)
        )
        self.assertIsNone(
            verify_account(self.db_path, "missing", "anything", self.hasher)
        )
        self.assertIsNone(
            verify_account(self.db_path, "broken", "anything", self.hasher)
        )

        class FalseReturningVerifier:
            def verify(self, _hash, _password):
                return False

        self.assertIsNone(
            verify_account(
                self.db_path,
                "alice",
                "correct-password",
                FalseReturningVerifier(),
            )
        )

    def test_missing_account_version_revokes_an_existing_session(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("DELETE FROM users WHERE username = 'alice'")
            conn.commit()
        self.assertIsNone(get_account_auth_version(self.db_path, "alice"))

    def test_legacy_schema_migration_is_transactional_and_idempotent(self):
        legacy_path = ROOT / f".test_account_auth_legacy_{uuid4().hex}.db"
        self.extra_paths.append(legacy_path)
        with closing(sqlite3.connect(legacy_path)) as conn:
            conn.execute(
                "CREATE TABLE users ("
                "username TEXT PRIMARY KEY, password_hash TEXT NOT NULL)"
            )
            conn.execute("INSERT INTO users VALUES ('legacy', 'hash')")
            conn.commit()

            init_account_auth_schema(conn)
            self.assertTrue(conn.in_transaction)
            conn.rollback()
            columns = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
            self.assertNotIn("auth_version", columns)

            init_account_auth_schema(conn)
            init_account_auth_schema(conn)
            conn.commit()
            columns = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
            version = conn.execute(
                "SELECT auth_version FROM users WHERE username = 'legacy'"
            ).fetchone()[0]
            self.assertIn("auth_version", columns)
            self.assertEqual(version, 1)

    def test_malformed_existing_auth_version_fails_closed(self):
        bad_path = ROOT / f".test_account_auth_bad_{uuid4().hex}.db"
        self.extra_paths.append(bad_path)
        with closing(sqlite3.connect(bad_path)) as conn:
            conn.execute(
                "CREATE TABLE users (username TEXT PRIMARY KEY, "
                "password_hash TEXT NOT NULL, auth_version INTEGER)"
            )
            conn.commit()
            with self.assertRaises(AccountAuthSchemaError):
                init_account_auth_schema(conn)


if __name__ == "__main__":
    unittest.main()
