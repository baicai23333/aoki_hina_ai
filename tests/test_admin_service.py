import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from admin_service import (
    AdminNotFoundError,
    AdminSchemaError,
    AdminValidationError,
    AuditEntry,
    DatabaseHealth,
    DeletedUserResult,
    TranslationBreakdown,
    UserSummary,
    clear_user_history,
    clear_user_memories,
    delete_user_account,
    get_database_health,
    get_overview,
    get_translation_breakdown,
    init_admin_schema,
    list_audit_entries,
    list_recent_messages,
    list_user_summaries,
    record_admin_action,
    replace_user_password_hash,
)


VALID_ARGON2_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$"
    "c29tZXNhbHQ$YW5lbmNvZGVkaGFzaA"
)
ROOT = Path(__file__).resolve().parents[1]


class AdminServiceTests(unittest.TestCase):
    def setUp(self):
        self.db_path = ROOT / f".test_admin_service_{uuid4().hex}.db"
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL
                );

                CREATE TABLE chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    japanese_content TEXT,
                    audio_path TEXT,
                    translation_status TEXT NOT NULL DEFAULT 'none' CHECK (
                        translation_status IN (
                            'validated', 'fixed', 'rejected', 'failed', 'none',
                            'legacy_unverified'
                        )
                    ),
                    translation_issue_code TEXT
                );

                CREATE TABLE user_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    category TEXT NOT NULL CHECK (
                        category IN (
                            'preferred_name', 'interest', 'goal',
                            'conversation_preference'
                        )
                    ),
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual_ui',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE,
                    UNIQUE (username, category, memory_key)
                );
                """
            )
            conn.executemany(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (
                    ("alice", "old-alice-hash"),
                    ("bob", "old-bob-hash"),
                    ("o'malley; DROP TABLE users;--", "old-injection-hash"),
                ),
            )
            conn.executemany(
                "INSERT INTO chat_history "
                "(username, type, content, timestamp, japanese_content, audio_path, "
                "translation_status, translation_issue_code) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        "alice",
                        "human",
                        "alice-question-1",
                        "2026-07-14T01:00:00Z",
                        None,
                        None,
                        "none",
                        None,
                    ),
                    (
                        "alice",
                        "ai",
                        "alice-answer-1",
                        "2026-07-14T01:00:01Z",
                        "alice-japanese-1",
                        "alice-1.wav",
                        "validated",
                        None,
                    ),
                    (
                        "alice",
                        "human",
                        "alice-question-2",
                        "2026-07-14T02:00:00Z",
                        None,
                        None,
                        "none",
                        None,
                    ),
                    (
                        "alice",
                        "ai",
                        "alice-answer-2",
                        "2026-07-14T02:00:01Z",
                        "alice-japanese-2",
                        None,
                        "fixed",
                        "polished",
                    ),
                    (
                        "bob",
                        "human",
                        "bob-question",
                        "2026-07-14T03:00:00Z",
                        None,
                        None,
                        "none",
                        None,
                    ),
                    (
                        "bob",
                        "ai",
                        "bob-answer",
                        "2026-07-14T03:00:01Z",
                        None,
                        None,
                        "rejected",
                        "review_rejected",
                    ),
                ),
            )
            conn.executemany(
                "INSERT INTO user_memories "
                "(username, category, memory_key, memory_value, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'manual_ui', ?, ?)",
                (
                    (
                        "alice",
                        "preferred_name",
                        "name",
                        "Alice",
                        "2026-07-14T01:00:00Z",
                        "2026-07-14T01:00:00Z",
                    ),
                    (
                        "bob",
                        "goal",
                        "goal",
                        "Practice",
                        "2026-07-14T01:00:00Z",
                        "2026-07-14T01:00:00Z",
                    ),
                ),
            )
            conn.commit()

    def tearDown(self):
        for suffix in ("", "-journal", "-shm", "-wal"):
            Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)

    def scalar(self, sql, parameters=()):
        with closing(sqlite3.connect(self.db_path)) as conn:
            return conn.execute(sql, parameters).fetchone()[0]

    def test_init_admin_schema_is_additive_idempotent_and_sets_busy_timeout(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            init_admin_schema(conn)
            init_admin_schema(conn)
            columns = [
                row[1] for row in conn.execute("PRAGMA table_info(admin_audit_log)")
            ]
            user_columns = [
                row[1] for row in conn.execute("PRAGMA table_info(users)")
            ]
            auth_versions = conn.execute(
                "SELECT DISTINCT auth_version FROM users"
            ).fetchall()
            busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            conn.commit()

        self.assertEqual(
            columns,
            ["id", "actor", "action", "target_username", "detail", "created_at"],
        )
        self.assertIn("auth_version", user_columns)
        self.assertEqual(auth_versions, [(1,)])
        self.assertEqual(busy_timeout, 5_000)
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM users"), 3)

    def test_overview_and_translation_breakdown_are_aggregated_without_content(self):
        overview = get_overview(self.db_path)
        breakdown = get_translation_breakdown(self.db_path)

        self.assertEqual(overview.total_users, 3)
        self.assertEqual(overview.total_messages, 6)
        self.assertEqual(overview.human_messages, 3)
        self.assertEqual(overview.ai_messages, 3)
        self.assertEqual(overview.total_memories, 2)
        self.assertEqual(overview.users_with_messages, 2)
        self.assertEqual(overview.latest_message_at, "2026-07-14T03:00:01Z")
        self.assertEqual(
            breakdown,
            TranslationBreakdown(
                validated=1,
                fixed=1,
                rejected=1,
                failed=0,
                none=0,
                legacy_unverified=0,
                total_ai_messages=3,
            ),
        )

    def test_user_summaries_support_stable_paging_and_literal_search(self):
        first_page = list_user_summaries(self.db_path, limit=1, offset=0)
        second_page = list_user_summaries(self.db_path, limit=1, offset=1)
        literal = list_user_summaries(
            self.db_path, search="o'malley; DROP TABLE users;--"
        )

        self.assertEqual(first_page, [UserSummary("bob", 2, 1, "2026-07-14T03:00:01Z")])
        self.assertEqual(
            second_page,
            [UserSummary("alice", 4, 1, "2026-07-14T02:00:01Z")],
        )
        self.assertEqual([item.username for item in literal], ["o'malley; DROP TABLE users;--"])
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM users"), 3)

    def test_recent_messages_hide_content_by_default_and_do_not_audit(self):
        messages = list_recent_messages(self.db_path, "alice", limit=2)

        self.assertEqual([message.id for message in messages], [4, 3])
        self.assertTrue(messages[0].has_japanese)
        self.assertFalse(messages[0].has_audio)
        self.assertIsNone(messages[0].content)
        self.assertIsNone(messages[0].japanese_content)
        self.assertEqual(list_audit_entries(self.db_path), [])

    def test_content_access_requires_actor_and_is_audited_atomically(self):
        with self.assertRaisesRegex(AdminValidationError, "actor is required"):
            list_recent_messages(
                self.db_path, "alice", include_content=True
            )

        messages = list_recent_messages(
            self.db_path,
            "alice",
            limit=2,
            offset=1,
            include_content=True,
            actor="admin",
        )
        entries = list_audit_entries(self.db_path)

        self.assertEqual([message.id for message in messages], [3, 2])
        self.assertEqual(messages[0].content, "alice-question-2")
        self.assertEqual(messages[1].japanese_content, "alice-japanese-1")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].actor, "admin")
        self.assertEqual(entries[0].action, "messages.content_viewed")
        self.assertEqual(entries[0].target_username, "alice")
        self.assertEqual(
            json.loads(entries[0].detail),
            {"limit": 2, "offset": 1, "returned_count": 2},
        )

    def test_unknown_user_content_access_is_not_audited(self):
        with self.assertRaises(AdminNotFoundError):
            list_recent_messages(
                self.db_path,
                "nobody",
                include_content=True,
                actor="admin",
            )
        self.assertEqual(list_audit_entries(self.db_path), [])

    def test_manual_audit_entries_are_parameterized_and_newest_first(self):
        first_id = record_admin_action(
            self.db_path,
            "admin'; DROP TABLE users;--",
            "dashboard.opened",
            target_username="alice",
            detail="'; DELETE FROM users;--",
        )
        second_id = record_admin_action(
            self.db_path, "admin", "settings.viewed"
        )
        entries = list_audit_entries(self.db_path, limit=1)

        self.assertGreater(second_id, first_id)
        self.assertEqual(entries, [AuditEntry(second_id, "admin", "settings.viewed", None, None, entries[0].created_at)])
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM users"), 3)

    def test_password_hash_replacement_is_scoped_and_never_logged(self):
        self.assertTrue(
            replace_user_password_hash(
                self.db_path, "admin", "alice", VALID_ARGON2_HASH
            )
        )

        self.assertEqual(
            self.scalar(
                "SELECT password_hash FROM users WHERE username = ?", ("alice",)
            ),
            VALID_ARGON2_HASH,
        )
        self.assertEqual(
            self.scalar(
                "SELECT auth_version FROM users WHERE username = ?", ("alice",)
            ),
            2,
        )
        self.assertEqual(
            self.scalar(
                "SELECT password_hash FROM users WHERE username = ?", ("bob",)
            ),
            "old-bob-hash",
        )
        entry = list_audit_entries(self.db_path)[0]
        self.assertEqual(entry.action, "user.password_hash_replaced")
        self.assertIsNone(entry.detail)
        self.assertNotIn(VALID_ARGON2_HASH, repr(entry))

        with self.assertRaises(AdminValidationError):
            replace_user_password_hash(self.db_path, "admin", "bob", "plaintext")
        with self.assertRaises(AdminNotFoundError):
            replace_user_password_hash(
                self.db_path, "admin", "nobody", VALID_ARGON2_HASH
            )

    def test_clear_history_and_memories_are_user_scoped_and_audited(self):
        self.assertEqual(clear_user_history(self.db_path, "admin", "alice"), 4)
        self.assertEqual(clear_user_memories(self.db_path, "admin", "alice"), 1)

        self.assertEqual(
            self.scalar(
                "SELECT COUNT(*) FROM chat_history WHERE username = ?", ("alice",)
            ),
            0,
        )
        self.assertEqual(
            self.scalar(
                "SELECT COUNT(*) FROM chat_history WHERE username = ?", ("bob",)
            ),
            2,
        )
        self.assertEqual(
            self.scalar(
                "SELECT COUNT(*) FROM user_memories WHERE username = ?", ("bob",)
            ),
            1,
        )
        actions = [entry.action for entry in list_audit_entries(self.db_path)]
        self.assertEqual(
            actions, ["user.memories_cleared", "user.history_cleared"]
        )

    def test_write_rolls_back_when_audit_insert_fails(self):
        record_admin_action(self.db_path, "admin", "schema.ready")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript(
                """
                CREATE TRIGGER reject_history_audit
                BEFORE INSERT ON admin_audit_log
                WHEN NEW.action = 'user.history_cleared'
                BEGIN
                    SELECT RAISE(ABORT, 'simulated audit failure');
                END;
                """
            )
            conn.commit()

        with self.assertRaises(sqlite3.IntegrityError):
            clear_user_history(self.db_path, "admin", "alice")

        self.assertEqual(
            self.scalar(
                "SELECT COUNT(*) FROM chat_history WHERE username = ?", ("alice",)
            ),
            4,
        )

    def test_delete_user_removes_all_scoped_data_but_preserves_audit(self):
        result = delete_user_account(self.db_path, "admin", "alice")

        self.assertEqual(
            result,
            DeletedUserResult(
                username="alice",
                messages_deleted=4,
                memories_deleted=1,
                user_deleted=True,
            ),
        )
        self.assertEqual(
            self.scalar("SELECT COUNT(*) FROM users WHERE username = ?", ("alice",)),
            0,
        )
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM users"), 2)
        entry = list_audit_entries(self.db_path)[0]
        self.assertEqual(entry.action, "user.account_deleted")
        self.assertEqual(entry.target_username, "alice")
        self.assertEqual(
            json.loads(entry.detail),
            {"memories_deleted": 1, "messages_deleted": 4},
        )

    def test_database_health_reports_integrity_and_schema(self):
        health = get_database_health(self.db_path)

        self.assertIsInstance(health, DatabaseHealth)
        self.assertTrue(health.ok)
        self.assertTrue(health.schema_ok)
        self.assertEqual(health.integrity_check, "ok")
        self.assertEqual(health.foreign_key_violations, 0)
        self.assertGreater(health.db_size_bytes, 0)

    def test_strict_argument_validation_happens_before_database_mutation(self):
        invalid_calls = (
            lambda: list_user_summaries(self.db_path, limit=True),
            lambda: list_user_summaries(self.db_path, limit=501),
            lambda: list_user_summaries(self.db_path, offset=-1),
            lambda: list_user_summaries(self.db_path, search="x" * 201),
            lambda: list_recent_messages(
                self.db_path, "alice", include_content="yes"
            ),
            lambda: record_admin_action(self.db_path, "admin", "INVALID ACTION"),
            lambda: record_admin_action(
                self.db_path, "admin", "valid.action", detail="x" * 4_001
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(AdminValidationError):
                    call()

        self.assertEqual(self.scalar("SELECT COUNT(*) FROM users"), 3)

    def test_incompatible_application_schema_fails_closed(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("ALTER TABLE users RENAME TO users_original")
            conn.execute(
                "CREATE TABLE users (username TEXT PRIMARY KEY)"
            )
            conn.commit()

        with self.assertRaisesRegex(AdminSchemaError, "password_hash"):
            get_overview(self.db_path)

    def test_downstream_schema_failure_rolls_back_admin_migrations(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("ALTER TABLE chat_history RENAME TO broken_chat_history")
            conn.execute("CREATE TABLE chat_history (id INTEGER PRIMARY KEY)")
            conn.commit()

        with self.assertRaisesRegex(AdminSchemaError, "chat_history missing"):
            get_overview(self.db_path)

        with closing(sqlite3.connect(self.db_path)) as conn:
            user_columns = [
                row[1] for row in conn.execute("PRAGMA table_info(users)")
            ]
            audit_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'admin_audit_log'"
            ).fetchone()
        self.assertNotIn("auth_version", user_columns)
        self.assertIsNone(audit_table)


if __name__ == "__main__":
    unittest.main()
