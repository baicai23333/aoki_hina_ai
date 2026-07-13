import sqlite3
import unittest
from contextlib import closing
from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import uuid4

from user_memory import (
    BUSY_TIMEOUT_MS,
    MAX_MEMORIES_PER_USER,
    MEMORY_CATEGORIES,
    UserMemoryLimitError,
    UserMemorySchemaError,
    UserMemoryValidationError,
    clear_memories,
    delete_memory,
    init_user_memory_schema,
    list_memories,
    upsert_memory,
)


ROOT = Path(__file__).resolve().parents[1]


class UserMemoryTests(unittest.TestCase):
    def setUp(self):
        # Direct workspace files work on restricted Windows hosts where Python's
        # TemporaryDirectory can create directories that SQLite cannot reopen.
        self.db_path = ROOT / f".test_user_memory_{uuid4().hex}.db"
        self.cleanup_paths = [self.db_path]
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "CREATE TABLE users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL)"
            )
            conn.execute(
                """
                CREATE TABLE chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                )
                """
            )
            conn.executemany(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (("alice", "hash-a"), ("bob", "hash-b")),
            )
            conn.execute(
                "INSERT INTO chat_history (username, type, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                ("alice", "human", "保留这条旧消息", "2026-07-14 00:00:00"),
            )
            conn.commit()

    def tearDown(self):
        for db_path in self.cleanup_paths:
            for suffix in ("", "-journal", "-shm", "-wal"):
                Path(f"{db_path}{suffix}").unlink(missing_ok=True)

    def init_schema(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            init_user_memory_schema(conn)
            conn.commit()

    def test_additive_migration_preserves_legacy_users_and_chat(self):
        self.init_schema()

        with closing(sqlite3.connect(self.db_path)) as conn:
            users = conn.execute(
                "SELECT username, password_hash FROM users ORDER BY username"
            ).fetchall()
            history = conn.execute(
                "SELECT username, type, content, timestamp FROM chat_history"
            ).fetchall()
            memory_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'user_memories'"
            ).fetchone()

        self.assertEqual(users, [("alice", "hash-a"), ("bob", "hash-b")])
        self.assertEqual(
            history,
            [("alice", "human", "保留这条旧消息", "2026-07-14 00:00:00")],
        )
        self.assertEqual(memory_table, ("user_memories",))

    def test_schema_initialization_is_idempotent_and_sets_pragmas(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            init_user_memory_schema(conn)
            init_user_memory_schema(conn)
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertEqual(
                conn.execute("PRAGMA busy_timeout").fetchone()[0], BUSY_TIMEOUT_MS
            )
            columns = [row[1] for row in conn.execute("PRAGMA table_info(user_memories)")]
            conn.commit()

        self.assertEqual(
            columns,
            [
                "id",
                "username",
                "category",
                "memory_key",
                "memory_value",
                "source",
                "created_at",
                "updated_at",
            ],
        )

    def test_upsert_reuses_identity_preserves_creation_and_is_frozen(self):
        first = upsert_memory(
            self.db_path, "alice", "preferred_name", "name", "  小爱  "
        )
        second = upsert_memory(
            self.db_path, "alice", "preferred_name", "name", "小葵"
        )

        self.assertEqual(first.id, second.id)
        self.assertEqual(first.created_at, second.created_at)
        self.assertEqual(second.memory_value, "小葵")
        self.assertEqual(second.source, "manual_ui")
        self.assertEqual(len(list_memories(self.db_path, "alice")), 1)
        with self.assertRaises(FrozenInstanceError):
            second.memory_value = "不能修改"

    def test_every_read_and_mutation_is_scoped_by_username(self):
        alice = upsert_memory(self.db_path, "alice", "interest", "instrument", "吉他")
        bob = upsert_memory(self.db_path, "bob", "interest", "instrument", "钢琴")

        self.assertEqual([item.id for item in list_memories(self.db_path, "alice")], [alice.id])
        self.assertEqual([item.id for item in list_memories(self.db_path, "bob")], [bob.id])
        self.assertFalse(delete_memory(self.db_path, "bob", alice.id))
        self.assertEqual(clear_memories(self.db_path, "alice"), 1)
        self.assertEqual(list_memories(self.db_path, "alice"), [])
        self.assertEqual([item.id for item in list_memories(self.db_path, "bob")], [bob.id])

    def test_validation_rejects_bad_categories_lengths_and_identifiers(self):
        self.assertEqual(
            MEMORY_CATEGORIES,
            ("preferred_name", "interest", "goal", "conversation_preference"),
        )
        invalid_calls = (
            lambda: upsert_memory(self.db_path, "", "interest", "topic", "music"),
            lambda: upsert_memory(self.db_path, "alice", "private_secret", "x", "y"),
            lambda: upsert_memory(self.db_path, "alice", "interest", " ", "music"),
            lambda: upsert_memory(self.db_path, "alice", "interest", "x" * 81, "music"),
            lambda: upsert_memory(self.db_path, "alice", "interest", "topic", " "),
            lambda: upsert_memory(self.db_path, "alice", "interest", "topic", "x" * 501),
            lambda: delete_memory(self.db_path, "alice", 0),
            lambda: delete_memory(self.db_path, "alice", True),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(UserMemoryValidationError):
                    call()

    def test_maximum_is_per_user_and_existing_keys_can_still_update(self):
        for index in range(MAX_MEMORIES_PER_USER):
            upsert_memory(
                self.db_path,
                "alice",
                "interest",
                f"topic-{index}",
                f"value-{index}",
            )

        self.assertEqual(len(list_memories(self.db_path, "alice")), MAX_MEMORIES_PER_USER)
        with self.assertRaises(UserMemoryLimitError):
            upsert_memory(self.db_path, "alice", "goal", "one-too-many", "blocked")

        updated = upsert_memory(
            self.db_path, "alice", "interest", "topic-0", "updated-at-limit"
        )
        self.assertEqual(updated.memory_value, "updated-at-limit")
        bob = upsert_memory(self.db_path, "bob", "goal", "first", "still allowed")
        self.assertEqual(bob.username, "bob")

    def test_sql_metacharacters_are_data_not_queries(self):
        username = "o'malley; DROP TABLE users;--"
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, "hash"),
            )
            conn.commit()

        memory = upsert_memory(
            self.db_path,
            username,
            "goal",
            "x'); DELETE FROM user_memories;--",
            "记住引号 ' 和分号 ; 都只是文本",
        )
        loaded = list_memories(self.db_path, username)

        self.assertEqual(loaded, [memory])
        with closing(sqlite3.connect(self.db_path)) as conn:
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM users").fetchone()[0], 3
            )

    def test_delete_is_a_hard_delete(self):
        memory = upsert_memory(self.db_path, "alice", "goal", "practice", "每天十分钟")
        self.assertTrue(delete_memory(self.db_path, "alice", memory.id))
        self.assertFalse(delete_memory(self.db_path, "alice", memory.id))

        with closing(sqlite3.connect(self.db_path)) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM user_memories WHERE id = ?", (memory.id,)
                ).fetchone()[0],
                0,
            )

    def test_unknown_username_cannot_create_a_memory(self):
        with self.assertRaisesRegex(UserMemoryValidationError, "does not exist"):
            upsert_memory(self.db_path, "nobody", "interest", "topic", "music")

    def test_incompatible_existing_table_fails_without_replacing_it(self):
        incompatible_path = ROOT / f".test_user_memory_incompatible_{uuid4().hex}.db"
        self.cleanup_paths.append(incompatible_path)
        with closing(sqlite3.connect(incompatible_path)) as conn:
            conn.execute(
                "CREATE TABLE users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE user_memories (id INTEGER PRIMARY KEY, username TEXT NOT NULL)"
            )
            conn.commit()

        with closing(sqlite3.connect(incompatible_path)) as conn:
            with self.assertRaisesRegex(UserMemorySchemaError, "Incompatible"):
                init_user_memory_schema(conn)

        with closing(sqlite3.connect(incompatible_path)) as conn:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(user_memories)")]
        self.assertEqual(columns, ["id", "username"])


if __name__ == "__main__":
    unittest.main()
