import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from chat_storage import (
    ChatStorageSchemaError,
    ChatStorageValidationError,
    StoredMessage,
    init_chat_storage_schema,
    list_messages,
    save_exchange,
    update_message_audio,
)


ROOT = Path(__file__).resolve().parents[1]


class ChatStorageTests(unittest.TestCase):
    def setUp(self):
        self.db_path = ROOT / f".test_chat_storage_{uuid4().hex}.db"
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
                    timestamp DATETIME NOT NULL,
                    japanese_content TEXT,
                    audio_path TEXT
                )
                """
            )
            conn.executemany(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (("alice", "hash-a"), ("bob", "hash-b")),
            )
            conn.commit()

    def tearDown(self):
        for db_path in self.cleanup_paths:
            for suffix in ("", "-journal", "-shm", "-wal"):
                Path(f"{db_path}{suffix}").unlink(missing_ok=True)

    def init_schema(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            init_chat_storage_schema(conn)
            conn.commit()

    def test_migration_marks_legacy_translation_and_is_idempotent(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executemany(
                "INSERT INTO chat_history "
                "(username, type, content, japanese_content, audio_path, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ("alice", "human", "旧问题", None, None, "2026-07-14 01:00:00"),
                    ("alice", "ai", "旧回答", "旧い翻訳", "old.wav", "2026-07-14 01:00:01"),
                    ("alice", "ai", "没有译文", None, None, "2026-07-14 01:00:02"),
                ),
            )
            conn.commit()
            init_chat_storage_schema(conn)
            init_chat_storage_schema(conn)
            conn.commit()
            columns = [row[1] for row in conn.execute("PRAGMA table_info(chat_history)")]

        messages = list_messages(self.db_path, "alice")

        self.assertEqual(columns.count("translation_status"), 1)
        self.assertEqual(columns.count("translation_issue_code"), 1)
        self.assertEqual(
            [message.translation_status for message in messages],
            ["none", "legacy_unverified", "none"],
        )
        self.assertEqual(messages[1].japanese_content, "旧い翻訳")
        self.assertEqual(messages[1].audio_path, "old.wav")
        self.assertTrue(all(isinstance(message, StoredMessage) for message in messages))

    def test_duplicate_chinese_replies_keep_distinct_ids_and_metadata(self):
        first_user_id, first_ai_id = save_exchange(
            self.db_path,
            "alice",
            "第一次",
            "相同中文回复",
            "最初の翻訳",
            "validated",
            None,
            "first.wav",
        )
        second_user_id, second_ai_id = save_exchange(
            self.db_path,
            "alice",
            "第二次",
            "相同中文回复",
            "修正した翻訳",
            "fixed",
            "polished",
            "second.wav",
        )

        messages = list_messages(self.db_path, "alice")
        ai_messages = [message for message in messages if message.type == "ai"]

        self.assertEqual(
            [message.id for message in messages],
            [first_user_id, first_ai_id, second_user_id, second_ai_id],
        )
        self.assertNotEqual(first_ai_id, second_ai_id)
        self.assertEqual([message.content for message in ai_messages], ["相同中文回复"] * 2)
        self.assertEqual(
            [message.japanese_content for message in ai_messages],
            ["最初の翻訳", "修正した翻訳"],
        )
        self.assertEqual(
            [message.audio_path for message in ai_messages],
            ["first.wav", "second.wav"],
        )

    def test_audio_update_is_scoped_by_username_type_and_translation_status(self):
        alice_user_id, alice_ai_id = save_exchange(
            self.db_path,
            "alice",
            "你好",
            "你好呀",
            "こんにちは",
            "validated",
            None,
            None,
        )
        _, bob_ai_id = save_exchange(
            self.db_path,
            "bob",
            "你好",
            "你好呀",
            "やあ",
            "fixed",
            None,
            None,
        )

        self.assertFalse(
            update_message_audio(self.db_path, "bob", alice_ai_id, "stolen.wav")
        )
        self.assertFalse(
            update_message_audio(self.db_path, "alice", alice_user_id, "human.wav")
        )
        self.assertFalse(
            update_message_audio(self.db_path, "alice", bob_ai_id, "other.wav")
        )
        self.assertTrue(
            update_message_audio(self.db_path, "alice", alice_ai_id, "alice.wav")
        )

        alice = {message.id: message for message in list_messages(self.db_path, "alice")}
        bob = {message.id: message for message in list_messages(self.db_path, "bob")}
        self.assertEqual(alice[alice_ai_id].audio_path, "alice.wav")
        self.assertIsNone(alice[alice_user_id].audio_path)
        self.assertIsNone(bob[bob_ai_id].audio_path)

    def test_exchange_rolls_back_user_row_when_ai_insert_fails(self):
        self.init_schema()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TRIGGER reject_ai_message
                BEFORE INSERT ON chat_history
                WHEN NEW.type = 'ai'
                BEGIN
                    SELECT RAISE(ABORT, 'simulated ai insert failure');
                END
                """
            )
            conn.commit()

        with self.assertRaises(sqlite3.IntegrityError):
            save_exchange(
                self.db_path,
                "alice",
                "不能留下半条记录",
                "这条会失败",
                "失敗します",
                "validated",
                None,
                None,
            )

        self.assertEqual(list_messages(self.db_path, "alice"), [])

    def test_list_order_uses_id_even_when_timestamps_disagree(self):
        first_ids = save_exchange(
            self.db_path,
            "alice",
            "先插入",
            "先回答",
            None,
            "none",
            None,
            None,
        )
        second_ids = save_exchange(
            self.db_path,
            "alice",
            "后插入",
            "后回答",
            None,
            "none",
            None,
            None,
        )
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE chat_history SET timestamp = '2099-01-01' WHERE id IN (?, ?)",
                first_ids,
            )
            conn.execute(
                "UPDATE chat_history SET timestamp = '2000-01-01' WHERE id IN (?, ?)",
                second_ids,
            )
            conn.commit()

        self.assertEqual(
            [message.id for message in list_messages(self.db_path, "alice")],
            [*first_ids, *second_ids],
        )

    def test_invalid_translation_is_not_stored_or_made_playable(self):
        invalid_ai_ids = []
        for status in ("rejected", "failed", "none", "legacy_unverified"):
            _, ai_id = save_exchange(
                self.db_path,
                "alice",
                f"问题-{status}",
                f"回答-{status}",
                "不应保存的译文",
                status,
                "not_approved",
                "must-not-play.wav",
            )
            invalid_ai_ids.append(ai_id)

        by_id = {message.id: message for message in list_messages(self.db_path, "alice")}
        for ai_id in invalid_ai_ids:
            with self.subTest(message_id=ai_id):
                self.assertIsNone(by_id[ai_id].japanese_content)
                self.assertIsNone(by_id[ai_id].audio_path)
                self.assertFalse(
                    update_message_audio(self.db_path, "alice", ai_id, "still-blocked.wav")
                )

        before = len(list_messages(self.db_path, "alice"))
        with self.assertRaises(ChatStorageValidationError):
            save_exchange(
                self.db_path,
                "alice",
                "问题",
                "回答",
                "訳文",
                "unreviewed",
                None,
                None,
            )
        self.assertEqual(len(list_messages(self.db_path, "alice")), before)

    def test_unknown_user_and_sql_metacharacters_are_safely_scoped(self):
        with self.assertRaisesRegex(ChatStorageValidationError, "does not exist"):
            save_exchange(
                self.db_path,
                "nobody",
                "问题",
                "回答",
                None,
                "none",
                None,
                None,
            )

        username = "o'malley; DROP TABLE users;--"
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, "hash"),
            )
            conn.commit()
        ids = save_exchange(
            self.db_path,
            username,
            "'; DELETE FROM chat_history;--",
            "引号和分号只是文本",
            None,
            "none",
            None,
            None,
        )

        self.assertEqual(
            [message.id for message in list_messages(self.db_path, username)],
            list(ids),
        )
        with closing(sqlite3.connect(self.db_path)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0], 3)

    def test_incompatible_legacy_schema_fails_before_partial_migration(self):
        incompatible_path = ROOT / f".test_chat_storage_incompatible_{uuid4().hex}.db"
        self.cleanup_paths.append(incompatible_path)
        with closing(sqlite3.connect(incompatible_path)) as conn:
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
            conn.commit()

        with closing(sqlite3.connect(incompatible_path)) as conn:
            with self.assertRaisesRegex(ChatStorageSchemaError, "required columns"):
                init_chat_storage_schema(conn)

        with closing(sqlite3.connect(incompatible_path)) as conn:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(chat_history)")]
        self.assertNotIn("translation_status", columns)
        self.assertNotIn("translation_issue_code", columns)

    def test_missing_users_username_fails_before_any_alter(self):
        incompatible_path = ROOT / f".test_chat_storage_bad_users_{uuid4().hex}.db"
        self.cleanup_paths.append(incompatible_path)
        with closing(sqlite3.connect(incompatible_path)) as conn:
            conn.execute("CREATE TABLE users (user_id TEXT PRIMARY KEY)")
            conn.execute(
                """
                CREATE TABLE chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    japanese_content TEXT,
                    audio_path TEXT
                )
                """
            )
            conn.commit()

            with self.assertRaisesRegex(ChatStorageSchemaError, "users.username"):
                init_chat_storage_schema(conn)
            conn.rollback()
            columns = [row[1] for row in conn.execute("PRAGMA table_info(chat_history)")]

        self.assertNotIn("translation_status", columns)
        self.assertNotIn("translation_issue_code", columns)

    def test_non_primary_message_id_fails_before_any_alter(self):
        incompatible_path = ROOT / f".test_chat_storage_bad_id_{uuid4().hex}.db"
        self.cleanup_paths.append(incompatible_path)
        with closing(sqlite3.connect(incompatible_path)) as conn:
            conn.execute("CREATE TABLE users (username TEXT PRIMARY KEY)")
            conn.execute(
                """
                CREATE TABLE chat_history (
                    id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    japanese_content TEXT,
                    audio_path TEXT
                )
                """
            )
            conn.commit()

            with self.assertRaisesRegex(ChatStorageSchemaError, "primary key"):
                init_chat_storage_schema(conn)
            conn.rollback()
            columns = [row[1] for row in conn.execute("PRAGMA table_info(chat_history)")]

        self.assertNotIn("translation_status", columns)
        self.assertNotIn("translation_issue_code", columns)

    def test_invalid_existing_status_fails_before_adding_issue_column(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("ALTER TABLE chat_history ADD COLUMN translation_status TEXT")
            conn.execute(
                "INSERT INTO chat_history "
                "(username, type, content, timestamp, japanese_content, audio_path, "
                "translation_status) VALUES (?, 'ai', ?, ?, ?, NULL, ?)",
                ("alice", "bad status", "2026-07-14 02:00:00", "翻訳", "bogus"),
            )
            conn.commit()

            with self.assertRaisesRegex(ChatStorageSchemaError, "invalid translation statuses"):
                init_chat_storage_schema(conn)
            conn.rollback()
            columns = [row[1] for row in conn.execute("PRAGMA table_info(chat_history)")]
            status = conn.execute(
                "SELECT translation_status FROM chat_history"
            ).fetchone()[0]

        self.assertNotIn("translation_issue_code", columns)
        self.assertEqual(status, "bogus")

    def test_update_failure_rolls_back_additive_columns(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO chat_history "
                "(username, type, content, timestamp, japanese_content, audio_path) "
                "VALUES (?, 'ai', ?, ?, ?, NULL)",
                ("alice", "legacy", "2026-07-14 03:00:00", "旧い翻訳"),
            )
            conn.execute(
                """
                CREATE TRIGGER reject_chat_history_update
                BEFORE UPDATE ON chat_history
                BEGIN
                    SELECT RAISE(ABORT, 'simulated migration update failure');
                END
                """
            )
            conn.commit()

            with self.assertRaises(sqlite3.IntegrityError):
                init_chat_storage_schema(conn)
            columns = [row[1] for row in conn.execute("PRAGMA table_info(chat_history)")]

        self.assertNotIn("translation_status", columns)
        self.assertNotIn("translation_issue_code", columns)


if __name__ == "__main__":
    unittest.main()
