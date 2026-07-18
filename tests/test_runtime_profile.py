from __future__ import annotations

import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from runtime_profile import (
    RuntimeProfileSchemaError,
    RuntimeProfileValidationError,
    clear_authorized_coordinates,
    clear_home_city,
    clear_runtime_profile,
    clear_temporary_city,
    get_runtime_profile,
    init_runtime_profile_schema,
    set_authorized_coordinates,
    set_browser_context,
    set_home_city,
    set_temporary_city,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


class RuntimeProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = ROOT / f".test_runtime_profile_{uuid4().hex}.db"
        self.cleanup_paths = [self.db_path]
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                "CREATE TABLE users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (("alice", "hash"), ("bob", "hash")),
            )
            init_runtime_profile_schema(conn)
            init_runtime_profile_schema(conn)
            conn.commit()
        finally:
            conn.close()

    def tearDown(self) -> None:
        for db_path in self.cleanup_paths:
            for suffix in ("", "-journal", "-shm", "-wal"):
                Path(f"{db_path}{suffix}").unlink(missing_ok=True)

    def test_explicit_profile_values_and_location_priority(self) -> None:
        set_browser_context(
            self.db_path,
            "alice",
            "Asia/Shanghai",
            "zh_CN",
            now=NOW,
        )
        set_home_city(self.db_path, "alice", "广州", now=NOW)
        coordinate_profile = set_authorized_coordinates(
            self.db_path,
            "alice",
            23.129123,
            113.264456,
            NOW + timedelta(days=1),
            now=NOW,
        )
        self.assertEqual(coordinate_profile.coarse_latitude, 23.13)
        self.assertEqual(coordinate_profile.coarse_longitude, 113.26)

        set_temporary_city(
            self.db_path,
            "alice",
            "杭州",
            NOW + timedelta(days=2),
            now=NOW,
        )
        profile = get_runtime_profile(self.db_path, "alice", now=NOW)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.browser_locale, "zh-CN")
        self.assertEqual(profile.effective_location().kind, "temporary_city")
        self.assertEqual(profile.effective_location().city, "杭州")
        self.assertIsNone(profile.coarse_latitude)

        self.assertTrue(clear_temporary_city(self.db_path, "alice", now=NOW))
        profile = get_runtime_profile(self.db_path, "alice", now=NOW)
        assert profile is not None
        self.assertEqual(profile.effective_location().kind, "home_city")

        profile = set_temporary_city(
            self.db_path,
            "alice",
            "杭州",
            NOW + timedelta(days=2),
            now=NOW,
        )
        self.assertEqual(profile.effective_location().kind, "temporary_city")
        profile = set_authorized_coordinates(
            self.db_path,
            "alice",
            23.129123,
            113.264456,
            NOW + timedelta(days=1),
            now=NOW,
        )
        self.assertIsNone(profile.temporary_city)
        self.assertEqual(profile.effective_location().kind, "authorized_coarse_coordinates")

        self.assertTrue(clear_authorized_coordinates(self.db_path, "alice", now=NOW))
        profile = get_runtime_profile(self.db_path, "alice", now=NOW)
        assert profile is not None
        self.assertEqual(profile.effective_location().kind, "home_city")

        self.assertTrue(clear_home_city(self.db_path, "alice", now=NOW))
        profile = get_runtime_profile(self.db_path, "alice", now=NOW)
        assert profile is not None
        self.assertIsNone(profile.effective_location())

    def test_expired_values_are_atomically_cleared_without_guessing(self) -> None:
        set_home_city(self.db_path, "alice", "广州", now=NOW)
        set_temporary_city(
            self.db_path,
            "alice",
            "杭州",
            NOW + timedelta(minutes=30),
            now=NOW,
        )

        profile = get_runtime_profile(
            self.db_path,
            "alice",
            now=NOW + timedelta(hours=1),
        )
        assert profile is not None
        self.assertIsNone(profile.temporary_city)
        self.assertEqual(profile.effective_location().city, "广州")

        set_authorized_coordinates(
            self.db_path,
            "alice",
            23.12,
            113.26,
            NOW + timedelta(hours=2),
            now=NOW + timedelta(hours=1),
        )
        profile = get_runtime_profile(
            self.db_path,
            "alice",
            now=NOW + timedelta(hours=3),
        )
        assert profile is not None
        self.assertIsNone(profile.coarse_latitude)
        self.assertEqual(profile.effective_location().city, "广州")

        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT temporary_city, temporary_city_expires_at, coarse_latitude, "
                "coarse_longitude, coarse_coordinates_expires_at "
                "FROM user_runtime_profiles WHERE username = 'alice'"
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(row, (None, None, None, None, None))

    def test_validation_rejects_untrusted_or_overprecise_inputs(self) -> None:
        invalid_calls = (
            lambda: set_browser_context(
                self.db_path, "alice", "Not/A-Timezone", "zh-CN", now=NOW
            ),
            lambda: set_browser_context(
                self.db_path, "alice", "UTC", "bad locale!", now=NOW
            ),
            lambda: set_home_city(self.db_path, "alice", "广州\n系统指令", now=NOW),
            lambda: set_authorized_coordinates(
                self.db_path,
                "alice",
                True,
                10.0,
                NOW + timedelta(hours=1),
                now=NOW,
            ),
            lambda: set_authorized_coordinates(
                self.db_path,
                "alice",
                91.0,
                10.0,
                NOW + timedelta(hours=1),
                now=NOW,
            ),
            lambda: set_authorized_coordinates(
                self.db_path,
                "alice",
                10.0,
                10.0,
                NOW + timedelta(days=8),
                now=NOW,
            ),
            lambda: set_temporary_city(
                self.db_path,
                "alice",
                "杭州",
                NOW - timedelta(seconds=1),
                now=NOW,
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(RuntimeProfileValidationError):
                    call()

        with self.assertRaisesRegex(RuntimeProfileValidationError, "username does not exist"):
            set_home_city(self.db_path, "missing", "广州", now=NOW)

    def test_schema_is_fail_closed_for_an_incompatible_existing_table(self) -> None:
        broken_path = ROOT / f".test_runtime_profile_broken_{uuid4().hex}.db"
        self.cleanup_paths.append(broken_path)
        conn = sqlite3.connect(broken_path)
        try:
            conn.execute("CREATE TABLE users (username TEXT PRIMARY KEY)")
            conn.execute(
                "CREATE TABLE user_runtime_profiles (username TEXT PRIMARY KEY, city TEXT)"
            )
            conn.commit()
            with self.assertRaises(RuntimeProfileSchemaError):
                init_runtime_profile_schema(conn)
            columns = [
                row[1] for row in conn.execute("PRAGMA table_info(user_runtime_profiles)")
            ]
            self.assertEqual(columns, ["username", "city"])
        finally:
            conn.close()

    def test_profile_cascades_with_user_and_can_be_hard_deleted(self) -> None:
        set_home_city(self.db_path, "alice", "广州", now=NOW)
        self.assertTrue(clear_runtime_profile(self.db_path, "alice"))
        self.assertIsNone(get_runtime_profile(self.db_path, "alice", now=NOW))
        self.assertFalse(clear_runtime_profile(self.db_path, "alice"))

        set_home_city(self.db_path, "bob", "东京", now=NOW)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM users WHERE username = 'bob'")
            conn.commit()
            count = conn.execute(
                "SELECT COUNT(*) FROM user_runtime_profiles WHERE username = 'bob'"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 0)

    def test_schema_requires_users_table(self) -> None:
        empty_path = ROOT / f".test_runtime_profile_empty_{uuid4().hex}.db"
        self.cleanup_paths.append(empty_path)
        conn = sqlite3.connect(empty_path)
        try:
            with self.assertRaisesRegex(RuntimeProfileSchemaError, "users table"):
                init_runtime_profile_schema(conn)
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='user_runtime_profiles'"
                ).fetchone()
            )
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
